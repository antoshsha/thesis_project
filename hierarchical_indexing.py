import os, time
from pathlib import Path
from dotenv import load_dotenv

from cohere import Client as CohereClient
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from classes import CohereRerankRetriever
from functions import (
    load_questions,
    sample,
    prompt_int,
    build_coarse_index,
    build_fine_index,
    create_retrieval_qa_chain,
)
from ragas import SingleTurnSample, EvaluationDataset, evaluate
import ragas.metrics as m

INDEX_DIR        = Path("faiss_index")
COARSE_INDEX_DIR = Path("faiss_index_coarse")
DATASET          = "dataset.json"

COARSE_CHUNK_SIZE    = 650
COARSE_CHUNK_OVERLAP = 200


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    if not openai_key or not cohere_key:
        raise RuntimeError("Missing keys in .env")

    all_qs = load_questions("questions.json")
    num = prompt_int("How many questions? ", default=None)
    seed = prompt_int("Seed? ", default=42)
    qs = sample(all_qs, num, seed)

    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o-mini", temperature=0.0)
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)
    co_client = CohereClient(api_key=cohere_key)
    coarse_vs = build_coarse_index(DATASET, COARSE_CHUNK_SIZE, COARSE_CHUNK_OVERLAP, embedder, COARSE_INDEX_DIR)

    experiments = [(500, 50)]

    SYSTEM_PREFIX = (
        "You are an expert in Ukrainian governmental services. "
        "If there is no relevant info in the retrieved docs, answer "
        "concisely from your knowledge without mentioning the source gap.\n\n"
    )

    for chunk_size, chunk_overlap in experiments:
        print(f"\n== Experiment: size={chunk_size}, overlap={chunk_overlap} ==")

        fine_dir = Path(f"faiss_index_{chunk_size}_{chunk_overlap}")
        vs = build_fine_index(DATASET, chunk_size, chunk_overlap, embedder, fine_dir)

        base_ret = vs.as_retriever(search_kwargs={"k": 16})
        reranked_ret = CohereRerankRetriever(base_ret, cohere_key, initial_k=8, final_k=1)
        qa_chain = create_retrieval_qa_chain(llm, reranked_ret)

        samples = []
        for i, qa_item in enumerate(qs, start=1):
            query = SYSTEM_PREFIX + qa_item["question"]
            out = qa_chain({"query": query})
            ans = out["result"].strip()
            ctxs = [d.page_content for d in out["source_documents"]]

            samples.append(SingleTurnSample(
                user_input=qa_item["question"],
                retrieved_contexts=ctxs,
                response=ans,
                reference=qa_item["answer"],
            ))

            print(f"[{i}/{len(qs)}] Q: {qa_item['question']}")
            print(f"→ {ans}\n")
            time.sleep(6)

        print("Running evaluation…")
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
