import os
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document

from classes import CohereRerankRetriever
from functions import (
    load_services,
    build_or_load_index,
    load_questions,
    sample,
    prompt_int,
)
from ragas import SingleTurnSample, EvaluationDataset, evaluate
import ragas.metrics as m

INDEX_DIR = Path("faiss_index")


class HypotheticalRetriever(BaseRetriever):
    """
    First retrieves k_real documents for the user's query, then
    asks the LLM to imagine k_hypo hypothetical sentences covering
    missing details, and finally retrieves one real doc per hypothetical.
    """
    base_retriever: BaseRetriever
    llm: ChatOpenAI
    k_real: int
    k_hypo: int

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: ChatOpenAI,
        k_real: int = 16,
        k_hypo: int = 8,
    ):
        super().__init__(
            base_retriever=base_retriever,
            llm=llm,
            k_real=k_real,
            k_hypo=k_hypo,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        real_docs = self.base_retriever.get_relevant_documents(query)[: self.k_real]

        context = "\n\n".join(d.page_content for d in real_docs)
        prompt = (
            f"Based on these excerpts:\n{context}\n\n"
            f"Generate {self.k_hypo} concise sentences that would cover any missing details "
            f"to fully answer: \"{query}\""
        )
        hypo_output = self.llm.predict(prompt)
        hypotheticals = [line.strip() for line in hypo_output.split("\n") if line.strip()]

        extra_docs: List[Document] = []
        for hypo in hypotheticals:
            hits = self.base_retriever.get_relevant_documents(hypo)[:1]
            extra_docs.extend(hits)

        return real_docs + extra_docs


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    co_key  = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    if not co_key:
        raise RuntimeError("COHERE_API_KEY missing in .env")

    all_qs = load_questions("questions.json")
    num = prompt_int("How many questions to test? (blank=all): ", default=None)
    seed = prompt_int("Random seed? (blank=42): ", default=42)
    sampled_qs = sample(all_qs, num, seed)

    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    experiments = [(500, 50)]
    SYSTEM_PREFIX = (
        "You are an expert in Ukrainian governmental services. "
        "If there is no relevant info in the retrieved docs, answer "
        "concisely from your knowledge without mentioning the source gap.\n\n"
    )

    for chunk_size, chunk_overlap in experiments:
        print(f"\n=== Experiment: chunk_size={chunk_size}, overlap={chunk_overlap} ===")

        index_dir = Path(f"faiss_index_{chunk_size}_{chunk_overlap}")
        if index_dir.exists():
            print(f"Found existing index at {index_dir}, skipping chunking.")
            vs = FAISS.load_local(str(index_dir), embeddings,
                                  allow_dangerous_deserialization=True)
        else:
            chunks = load_services("dataset.json", chunk_size, chunk_overlap)
            vs = build_or_load_index(chunks, embeddings, index_dir)
        base_ret = vs.as_retriever(search_kwargs={"k": 16})
        hypo_ret = HypotheticalRetriever(
            base_retriever=base_ret,
            llm=llm,
            k_real=1,
            k_hypo=7,
        )
        wrapped_ret = CohereRerankRetriever(
            base_retriever=hypo_ret,
            cohere_api_key=co_key,
            model="rerank-v3.5",
            initial_k=8,
            final_k=1,
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=wrapped_ret,
            return_source_documents=True,
        )
        samples: List[SingleTurnSample] = []
        for i, qa_item in enumerate(sampled_qs, start=1):
            inp = SYSTEM_PREFIX + qa_item["question"]
            out = qa({"query": inp})
            ans = out["result"].strip()
            ctxs = [d.page_content for d in out["source_documents"]]
            samples.append(SingleTurnSample(
                user_input=qa_item["question"],
                retrieved_contexts=ctxs,
                response=ans,
                reference=qa_item["answer"],
            ))
            time.sleep(6)
            print(f"[{i}/{len(sampled_qs)}] Q: {qa_item['question']}")
            print(f"→ A: {ans}")
            print(f"✓ ref: {qa_item['answer']}\n")
        ds = EvaluationDataset(samples=samples)
        print("Running Ragas evaluation…")
        metrics = [
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
        ]
        result = evaluate(dataset=ds, metrics=metrics, llm=llm)
        print(result)


if __name__ == "__main__":
    main()
