import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from classes import CohereRerankRetriever, PineconeRerankRetriever
from functions import (
    load_services,
    build_or_load_index,
    load_questions,
    sample,
    prompt_int,
    throttle,
)
from ragas import SingleTurnSample, EvaluationDataset, evaluate
import ragas.metrics as m

DATASET = "dataset.json"
INDEX_DIR = Path("faiss_index_500_50")
SYSTEM = (
    "You are an expert in Ukrainian governmental services. "
    "Answer *only* about those services, concisely, and if you draw on your own knowledge, "
    "never mention missing sources.\n\n"
)


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    pine_key = os.getenv("PINECONE_API_KEY")
    pine_env = os.getenv("PINECONE_ENV")
    pine_index = os.getenv("PINECONE_INDEX_NAME")

    if not all([openai_key, cohere_key, pine_key, pine_env, pine_index]):
        raise RuntimeError("Set OPENAI_API_KEY, COHERE_API_KEY, PINECONE_API_KEY, PINECONE_ENV & PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=pine_key, environment=pine_env)
    pc_idx = pc.Index(name=pine_index)
    all_qs = load_questions("questions.json")
    num = prompt_int("How many questions? (blank=all): ", default=None)
    seed = prompt_int("Random seed? (blank=42): ", default=42)
    qs = sample(all_qs, num, seed)
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    chunks = load_services(DATASET, chunk_size=500, chunk_overlap=50)
    vs = build_or_load_index(chunks, embeddings, INDEX_DIR)
    co_retriever = CohereRerankRetriever(
        base_retriever=vs.as_retriever(search_kwargs={"k": 20}),
        cohere_api_key=cohere_key,
        model="rerank-v3.5",
        initial_k=24,
        final_k=8,
    )
    pc_retriever = PineconeRerankRetriever(
        retriever=co_retriever,
        pc_index=pc_idx,
        model="bge-reranker-v2-m3",
        top_k=8,
        top_n=4,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=pc_retriever,
        return_source_documents=True,
    )
    samples: List[SingleTurnSample] = []
    for i, qa_item in enumerate(qs, start=1):
        prompt_text = SYSTEM + qa_item["question"]
        out = qa({"query": prompt_text})
        ans = out["result"].strip()
        ctxs = [d.page_content for d in out["source_documents"]]
        print(f"[{i}/{len(qs)}] Q: {qa_item['question']}")
        print(f"→ {ans}\n")
        samples.append(SingleTurnSample(
            user_input=qa_item["question"],
            retrieved_contexts=ctxs,
            response=ans,
            reference=qa_item["answer"],
        ))
        throttle()
    print("Running Ragas evaluation…")
    ds = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=ds,
        metrics=[
            m.answer_relevancy, m.answer_similarity, m.answer_correctness,
            m.ResponseRelevancy(), m.FactualCorrectness(), m.SemanticSimilarity(),
            m.NonLLMStringSimilarity(), m.BleuScore(), m.RougeScore(), m.AnswerAccuracy(),
        ],
        llm=llm,
    )
    print(result)


if __name__ == "__main__":
    main()
