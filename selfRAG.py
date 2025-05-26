import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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

retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "Given the user question:\n"
        "{query}\n\n"
        "Reply **only** 'Yes' or 'No'—do I need to retrieve documents to answer?"
    ),
)

relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "Given the question:\n{query}\n\n"
        "and this context:\n{context}\n\n"
        "Reply **only** 'Relevant' or 'Irrelevant'."
    ),
)

generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "Given the question:\n{query}\n\n"
        "and this context (if any):\n{context}\n\n"
        "Generate a concise answer in Ukrainian."
    ),
)

support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template=(
        "Given this generated answer:\n{response}\n\n"
        "and the context:\n{context}\n\n"
        "Reply **only** 'Fully supported', 'Partially supported', or 'No support'."
    ),
)

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template=(
        "Given the question:\n{query}\n\n"
        "and the generated answer:\n{response}\n\n"
        "Rate its utility from 1 to 5 (reply only the digit)."
    ),
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
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4.1-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    chunks = load_services("dataset.json", chunk_size=500, chunk_overlap=50)
    vs = build_or_load_index(chunks, embeddings, INDEX_DIR)
    retrieval_chain = LLMChain(llm=llm, prompt=retrieval_prompt)
    relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
    generation_chain = LLMChain(llm=llm, prompt=generation_prompt)
    support_chain = LLMChain(llm=llm, prompt=support_prompt)
    utility_chain = LLMChain(llm=llm, prompt=utility_prompt)

    SYSTEM = (
        "You are an expert in Ukrainian governmental services. "
        "Answer *only* about those services, concisely, "
        "and never mention missing sources.\n\n"
    )

    samples: List[SingleTurnSample] = []
    for i, qa_item in enumerate(qs, start=1):
        q = qa_item["question"]
        print(f"\n[{i}/{len(qs)}] Question: {q}")
        print(f"✓ Reference: {qa_item['answer']}\n")
        dec = retrieval_chain.run(query=q).strip().lower()
        print("Retrieval needed?", dec)

        if dec == "yes":
            docs = vs.similarity_search(q, k=4)
            print(f"→ Retrieved {len(docs)} documents")
            rels = []
            for idx, d in enumerate(docs, start=1):
                r = relevance_chain.run(query=q, context=d.page_content).strip().lower()
                print(f" Doc {idx} relevance: {r}")
                if r == "relevant":
                    rels.append(d)

            if not rels:
                resp = generation_chain.run(query=q, context="").strip()
                used_ctx: List[str] = []
                print("No relevant contexts → generated without retrieval")
            else:
                candidates = []
                for idx, d in enumerate(rels, start=1):
                    g = generation_chain.run(query=q, context=d.page_content).strip()
                    s = support_chain.run(response=g, context=d.page_content).strip().lower()
                    u_str = utility_chain.run(query=q, response=g).strip()
                    u = int(u_str) if u_str.isdigit() else 0
                    print(f" Candidate {idx} → support={s}, utility={u}")
                    score = (1 if s == "fully supported" else 0) * 10 + u
                    candidates.append((g, score, d.page_content))
                resp, _, best_ctx = max(candidates, key=lambda x: x[1])
                used_ctx = [best_ctx]
                print("Selected best candidate based on support+utility")
        else:
            resp = generation_chain.run(query=q, context="").strip()
            used_ctx = []
            print("Generated without retrieval")
        print("→ Final answer:", resp)
        samples.append(
            SingleTurnSample(
                user_input=q,
                retrieved_contexts=used_ctx,
                response=resp,
                reference=qa_item["answer"],
            )
        )
        throttle()
    print("\nRunning Ragas evaluation")
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
