import os
import json
import random
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from cohere import ClientV2 as CohereClient
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragas import SingleTurnSample, EvaluationDataset, evaluate
import ragas.metrics as m

INDEX_DIR = Path("faiss_index")


class CohereRerankRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    co: CohereClient
    model: str
    initial_k: int
    final_k: int

    def __init__(
        self,
        base_retriever: BaseRetriever,
        cohere_api_key: str,
        model: str = "rerank-v3.5",
        initial_k: int = 16,
        final_k: int = 4,
    ):
        co = CohereClient(api_key=cohere_api_key)
        super().__init__(
            base_retriever=base_retriever,
            co=co,
            model=model,
            initial_k=initial_k,
            final_k=final_k,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        candidates = self.base_retriever.get_relevant_documents(query)[: self.initial_k]
        texts = [d.page_content for d in candidates]
        resp = self.co.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=self.final_k,
        )
        return [candidates[r.index] for r in resp.results]


retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "Given the question:\n{query}\n\n"
        "Do I need to retrieve documents to answer it? Reply only 'Yes' or 'No'."
    ),
)

relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "Given the question:\n{query}\n\n"
        "and this context:\n{context}\n\n"
        "Is the context Relevant or Irrelevant? Reply only 'Relevant' or 'Irrelevant'."
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
        "Given the answer:\n{response}\n\n"
        "and the context:\n{context}\n\n"
        "Reply 'Fully supported', 'Partially supported', or 'No support'."
    ),
)

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template=(
        "Given the question:\n{query}\n\n"
        "and the answer:\n{response}\n\n"
        "Rate its utility from 1 to 5 (reply only the digit)."
    ),
)


def load_services(path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    raw = json.load(open(path, encoding="utf-8"))
    services = raw.get("entries") if isinstance(raw, dict) else raw
    docs = []
    for svc in services:
        text = "\n".join([
            f"Title: {svc.get('name','')}",
            f"Keyword: {svc.get('keyword','')}",
            f"Refusal grounds: {svc.get('refusal_grounds','')}",
            f"Applicant type: {svc.get('applicant_type','')}",
            f"Regulatory documents: {svc.get('regulatory_documents','')}",
            f"Life events: {svc.get('events','')}",
            f"Appeal persons: {svc.get('refusal_appeal_person','')}",
            f"Appealable in court: {svc.get('is_appealed_in_court','')}",
            f"Produces: {svc.get('produces','')}",
            f"Application ways: {svc.get('application_ways','')}",
            f"Receiving ways: {svc.get('receiving_ways','')}",
            f"Processing durations: {svc.get('processing_durations','')}",
            f"Costs: {svc.get('costs','')}",
            f"Description: {svc.get('short_description_plain','')}",
            f"Required documents: {svc.get('input','')}",
        ])
        docs.append(Document(page_content=text, metadata={"id": svc.get("identifier")}))
    print("First raw service document:\n", docs[0])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    return splitter.split_documents(docs)


def build_or_load_index(chunks, embeddings, index_dir: Path):
    if index_dir.exists():
        return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))
    return vs


def load_questions(path="questions.json"):
    return json.load(open(path, encoding="utf-8"))


def sample(lst, n, seed=42):
    r = random.Random(seed)
    return lst if n is None or n >= len(lst) else r.sample(lst, n)


def prompt_int(msg, default=None):
    s = input(msg).strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        print("Enter an integer or blank.")
        return prompt_int(msg, default)


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
    retrieval_chain = LLMChain(llm=llm, prompt=retrieval_prompt)
    relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
    generation_chain = LLMChain(llm=llm, prompt=generation_prompt)
    support_chain = LLMChain(llm=llm, prompt=support_prompt)
    utility_chain = LLMChain(llm=llm, prompt=utility_prompt)
    chunks = load_services("dataset.json", chunk_size=500, chunk_overlap=50)
    vs = build_or_load_index(chunks, embeddings, INDEX_DIR)
    base_ret = vs.as_retriever(search_kwargs={"k": 16})
    reranker = CohereRerankRetriever(
        base_retriever=base_ret,
        cohere_api_key=cohere_key,
        model="rerank-v3.5",
        initial_k=16,
        final_k=8,
    )

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
            docs = reranker.get_relevant_documents(q)
            print(f"→ Retrieved {len(docs)} docs after reranking")
            rels = []
            for idx, d in enumerate(docs, start=1):
                r = relevance_chain.run(query=q, context=d.page_content).strip().lower()
                print(f" Doc {idx} relevance: {r}")
                if r == "relevant":
                    rels.append(d)
            if not rels:
                resp = generation_chain.run(query=q, context="").strip()
                used_ctx = []
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
                print("Selected best candidate")
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
        time.sleep(6)
    print("\nRunning Ragas evaluation…")
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
