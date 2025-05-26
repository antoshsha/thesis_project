import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from classes import HypotheticalRetriever, CohereRerankRetriever
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
INDEX_PREFIX = "faiss_index"
COARSE_CHUNK_SIZE = 650
COARSE_CHUNK_OVERLAP = 200

SYSTEM_PREFIX = (
    "You are an expert in Ukrainian governmental services. "
    "If there is no relevant info in the retrieved docs, answer "
    "concisely without mentioning missing sources.\n\n"
)

EXPERIMENTS = [(500, 50)]


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    if not openai_key or not cohere_key:
        raise RuntimeError("Missing OPENAI_API_KEY or COHERE_API_KEY")
    all_qs = load_questions("questions.json")
    num = prompt_int("How many questions? (blank=all) ", default=None)
    seed = prompt_int("Random seed? (blank=42) ", default=42)
    qs = sample(all_qs, num, seed)
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    for chunk_size, chunk_overlap in EXPERIMENTS:
        print(f"\n=== Experiment: size={chunk_size}, overlap={chunk_overlap} ===")
        idx_dir = Path(f"{INDEX_PREFIX}_{chunk_size}_{chunk_overlap}")
        chunks = load_services(DATASET, chunk_size, chunk_overlap)
        vs = build_or_load_index(chunks, embeddings, idx_dir)
        base_ret = vs.as_retriever(search_kwargs={"k": 16})
        hypo_ret = HypotheticalRetriever(
            base_retriever=base_ret,
            llm=llm,
            embeddings=embeddings,
            k_real=4,
            k_hypo=4,
        )
        reranked_ret = CohereRerankRetriever(
            base_retriever=hypo_ret,
            cohere_api_key=cohere_key,
            initial_k=8,
            final_k=1,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=reranked_ret,
            return_source_documents=True,
        )
        samples: List[SingleTurnSample] = []
        for i, qa_item in enumerate(qs, start=1):
            inp = SYSTEM_PREFIX + qa_item["question"]
            out = qa_chain({"query": inp})
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
                m.answer_relevancy,
                m.answer_similarity,
                m.answer_correctness,
                m.ResponseRelevancy(),
                m.FactualCorrectness(),
                m.SemanticSimilarity(),
                m.NonLLMStringSimilarity(),
                m.BleuScore(),
                m.RougeScore(),
                m.AnswerAccuracy(),
            ],
            llm=llm,
        )
        print(result)


if __name__ == "__main__":
    main()
