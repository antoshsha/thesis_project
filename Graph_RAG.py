import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from classes import CohereRerankRetriever, KnowledgeGraph, GraphRAGEngine
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

INDEX_DIR = Path("faiss_index")
QUESTIONS = Path("questions.json")
DATASET   = Path("dataset.json")


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    if not openai_key or not cohere_key:
        raise RuntimeError("Missing OPENAI_API_KEY or COHERE_API_KEY in .env")

    all_qs = load_questions(str(QUESTIONS))
    num = prompt_int("How many questions to test? (blank=all): ", default=None)
    seed = prompt_int("Random seed? (blank=42): ", default=42)
    qs = sample(all_qs, num, seed)

    llm = ChatOpenAI(openai_api_key=openai_key,
                     model_name="gpt-4o-mini",
                     temperature=0.0)
    embedder = OpenAIEmbeddings(openai_api_key=openai_key)
    chunks = load_services(str(DATASET), chunk_size=500, chunk_overlap=50)
    vs = build_or_load_index(chunks, embedder, INDEX_DIR)
    kg = KnowledgeGraph(threshold=0.92)
    kg.load_or_build(chunks, embedder)
    base_ret = vs.as_retriever(search_kwargs={"k": 16})
    reranker = CohereRerankRetriever(base_ret, cohere_key,
                                     initial_k=8, final_k=3)
    engine = GraphRAGEngine(vs, kg, llm, reranker=reranker, hops=1)

    samples = []
    for i, qa in enumerate(qs, start=1):
        q, ref = qa["question"], qa["answer"]
        print(f"[{i}/{len(qs)}] Q: {q}")
        ans, ctxs = engine.query(q, k=3)
        print("→", ans, "\n")
        samples.append(SingleTurnSample(
            user_input=q,
            retrieved_contexts=ctxs,
            response=ans,
            reference=ref,
        ))
        throttle()

    print("Running evaluation…")
    ds = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=ds,
        metrics=[
            m.answer_relevancy, m.answer_similarity, m.answer_correctness,
            m.ResponseRelevancy(), m.FactualCorrectness(), m.SemanticSimilarity(),
            m.NonLLMStringSimilarity(), m.BleuScore(), m.RougeScore(),
            m.AnswerAccuracy(),
        ],
        llm=llm,
    )
    print(result)


if __name__ == "__main__":
    main()
