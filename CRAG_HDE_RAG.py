import os
from pathlib import Path

from dotenv import load_dotenv
from cohere import Client as CohereClient
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from classes import HypotheticalRetriever
from functions import (
    load_services,
    build_or_load_index,
    load_questions,
    sample,
    prompt_int,
    create_rewrite_chain,
    create_summarize_chain,
    get_search_tool,
    throttle,
)
from ragas import SingleTurnSample, EvaluationDataset, evaluate
import ragas.metrics as m

INDEX_DIR = Path("faiss_index")
SYSTEM_PREFIX = (
    "You are an expert in Ukrainian governmental services. "
    "If there is no relevant info in the retrieved docs, answer "
    "concisely from your knowledge without mentioning the source gap.\n\n"
)


def main():
    load_dotenv()
    api_key, co_key = os.getenv("OPENAI_API_KEY"), os.getenv("COHERE_API_KEY")
    if not api_key or not co_key:
        raise RuntimeError("OPENAI_API_KEY and COHERE_API_KEY must be set in .env")

    questions = sample(
        load_questions("questions.json"),
        prompt_int("How many questions to test? (blank=all): ", None),
        prompt_int("Random seed? (blank=42): ", 42),
    )

    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4.1", temperature=0.0)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    co_client = CohereClient(api_key=co_key)

    rewrite_chain = create_rewrite_chain(llm)
    summarize_chain = create_summarize_chain(llm)
    search_tool = get_search_tool()

    chunks = load_services("dataset.json", chunk_size=500, chunk_overlap=50)
    vs = build_or_load_index(chunks, embeddings, INDEX_DIR)

    samples = []
    for idx, qa in enumerate(questions, start=1):
        q, ref = qa["question"], qa["answer"]
        print(f"\n[{idx}/{len(questions)}] Q: {q}")

        base_ret = vs.as_retriever(search_kwargs={"k": 8})
        hypo_ret = HypotheticalRetriever(
            base_retriever=base_ret,
            llm=llm,
            embeddings=embeddings,
            k_real=5,
            k_hypo=3,
        )
        cands = hypo_ret.get_relevant_documents(q)
        texts = [d.page_content for d in cands]
        rerank = co_client.rerank(
            model="rerank-v3.5",
            query=q,
            documents=texts,
            top_n=1,
        ).results[0]
        score, chunk = rerank.relevance_score, texts[rerank.index]
        print(f"CRAG-HDE: best chunk idx={rerank.index}, score={score:.2f}")

        if score > 0.7:
            context = chunk
        elif score < 0.3:
            rew = rewrite_chain.run(query=q).rewritten
            web = search_tool.run(rew) or "No web results found."
            context = summarize_chain.run(text=web).summary
        else:
            rew = rewrite_chain.run(query=q).rewritten
            web = search_tool.run(rew)
            summary = summarize_chain.run(text=web or "").summary
            context = chunk + "\n\n" + summary

        prompt = (
            f"{SYSTEM_PREFIX}"
            f"Question: {q}\n"
            f"Context:\n{context}\n\n"
            "Answer concisely in Ukrainian:"
        )
        answer = llm.predict(prompt).strip()
        print(f"→ A: {answer}\n✓ ref: {ref}")

        samples.append(SingleTurnSample(
            user_input=q,
            retrieved_contexts=[context],
            response=answer,
            reference=ref,
        ))

        throttle()

    print("\n Running Ragas evaluation…")
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
