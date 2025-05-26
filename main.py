import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from classes import CohereRerankRetriever
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
INDEX_DIR = Path("faiss_index")
EXPERIMENTS = [(500, 50)]

SYSTEM_PREFIX = (
    "You are an expert in Ukrainian governmental services. "
    "If there is no relevant info in the retrieved docs, answer "
    "concisely from your knowledge without mentioning any missing sources.\n\n"
)


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    if not openai_key or not cohere_key:
        raise RuntimeError("OPENAI_API_KEY and COHERE_API_KEY must be set in .env")
    all_qs = load_questions("questions.json")
    num = prompt_int("How many questions to test? (blank=all): ", default=None)
    seed = prompt_int("Random seed? (blank=42): ", default=42)
    qs = sample(all_qs, num, seed)
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    for chunk_size, chunk_overlap in EXPERIMENTS:
        print(f"\n=== Experiment: chunk_size={chunk_size}, overlap={chunk_overlap} ===")
        idx_dir = INDEX_DIR.parent / f"faiss_index_{chunk_size}_{chunk_overlap}"
        chunks = load_services(DATASET, chunk_size, chunk_overlap)
        vs = build_or_load_index(chunks, embeddings, idx_dir)
        base_ret = vs.as_retriever(search_kwargs={"k": 16})
        wrapped_ret = CohereRerankRetriever(
            base_retriever=base_ret,
            cohere_api_key=cohere_key,
            model="rerank-v3.5",
            initial_k=8,
            final_k=1,
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=wrapped_ret,
            return_source_documents=True,
        )
        samples: List[SingleTurnSample] = []
        for i, qa_item in enumerate(qs, start=1):
            query = SYSTEM_PREFIX + qa_item["question"]
            out = qa_chain({"query": query})
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
