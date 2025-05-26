import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document
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
import ragas.metrics as met


class HypotheticalRetriever(BaseRetriever):
    """
    First retrieves k_real docs for the query, then asks the LLM
    to generate k_hypo “hypothetical” sentences filling missing gaps,
    then retrieves one real doc per hypothetical.
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
            f"Generate {self.k_hypo} concise sentences that cover any missing details "
            f"to fully answer: \"{query}\""
        )
        hypo_output = self.llm.predict(prompt)
        hypotheticals = [ln.strip() for ln in hypo_output.splitlines() if ln.strip()]

        extra = []
        for hypo in hypotheticals:
            hit = self.base_retriever.get_relevant_documents(hypo)[:1]
            extra.extend(hit)

        return real_docs + extra


def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    if not openai_key or not cohere_key:
        raise RuntimeError("Missing OPENAI_API_KEY or COHERE_API_KEY in .env")

    all_qs = load_questions("questions.json")
    num = prompt_int("How many questions? (blank=all) ", default=None)
    seed = prompt_int("Random seed? (blank=42) ", default=42)
    sampled_qs = sample(all_qs, num, seed)

    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4o-mini", temperature=0.0)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    SYSTEM = (
        "You are an expert in Ukrainian governmental services. "
        "If there is no relevant info in the retrieved docs, answer concisely "
        "from your knowledge without mentioning any missing sources.\n\n"
    )

    experiments = [(500, 50)]
    for chunk_size, chunk_overlap in experiments:
        print(f"\n=== Experiment: size={chunk_size}, overlap={chunk_overlap} ===")

        idx_dir = Path(f"faiss_index_{chunk_size}_{chunk_overlap}")
        chunks = load_services("dataset.json", chunk_size, chunk_overlap)
        vs = build_or_load_index(chunks, embeddings, idx_dir)

        base_ret = vs.as_retriever(search_kwargs={"k": 1})
        hypo_ret = HypotheticalRetriever(base_retriever=base_ret, llm=llm, k_real=1, k_hypo=7)
        wrapped_ret = CohereRerankRetriever(
            base_retriever=hypo_ret,
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
        for i, qa_item in enumerate(sampled_qs, start=1):
            q = qa_item["question"]
            print(f"\n[{i}/{len(sampled_qs)}] Question: {q}")
            docs1 = base_ret.get_relevant_documents(q)
            ctx1 = "\n\n".join(d.page_content for d in docs1)
            prompt1 = (
                f"{SYSTEM}Original question: {q}\n\nContext:\n{ctx1}\n\n"
                "Instructions:\n"
                "- Answer only the question.\n"
                "- Do not invent new facts.\n"
                "1) Provide an extremely concise answer in Ukrainian.\n"
                "2) Return JSON with fields “answer” and “gaps”.\n"
            )
            partial = llm.predict(prompt1)
            print("Partial model output:\n", partial)
            ans1, gaps = partial, []
            mobj = re.search(r"\{.*\}", partial, flags=re.DOTALL)
            if mobj:
                try:
                    jd = json.loads(mobj.group())
                    ans1 = jd.get("answer","")
                    gaps = jd.get("gaps",[])
                except json.JSONDecodeError:
                    pass
            if gaps:
                q2 = llm.predict(
                    f"Original question: {q}\nMissing information: {gaps}\n"
                    "Formulate a single follow-up question to obtain this information."
                ).strip()
                print("Follow-up query:", q2)

                docs2 = base_ret.get_relevant_documents(q2)[:5]
                ctx2  = "\n\n".join(d.page_content for d in docs2)
                final_prompt = (
                    f"{SYSTEM}Context1:\n{ctx1}\n\nContext2:\n{ctx2}\n\n"
                    "Using all context, produce the final concise answer in Ukrainian.\n"
                    "If there is no relevant info, answer concisely without mentioning gaps.\n"
                    "THE FINAL ANSWER MUST BE IN UKRAINIAN ONLY.\n\n"
                )
                final = llm.predict(final_prompt).strip()
                retrieved = docs1 + docs2
            else:
                final = ans1.strip()
                retrieved = docs1

            print("→ Final answer:", final)
            samples.append(SingleTurnSample(
                user_input=q,
                retrieved_contexts=[d.page_content for d in retrieved],
                response=final,
                reference=qa_item["answer"],
            ))
            throttle()

        print("\nRunning Ragas evaluation…")
        ds = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=ds,
            metrics=[
                met.answer_relevancy,
                met.answer_similarity,
                met.answer_correctness,
                met.ResponseRelevancy(),
                met.FactualCorrectness(),
                met.SemanticSimilarity(),
                met.NonLLMStringSimilarity(),
                met.BleuScore(),
                met.RougeScore(),
                met.AnswerAccuracy(),
            ],
            llm=llm,
        )
        print(result)


if __name__ == "__main__":
    main()
