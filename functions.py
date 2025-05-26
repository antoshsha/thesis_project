# functions.py

import json
import random
import time
from typing import List
from langchain.schema import BaseRetriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchResults
from langchain.output_parsers import PydanticOutputParser
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import Optional

from classes import RewriteOutput, SummarizeOutput


_rewrite_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "Rewrite this user query for a web search; make it concise and keyword-rich while keeping the context.\n\n"
        "**Output must be valid JSON** with a single field `rewritten`.\n\n"
        "User query:\n"
        "{query}\n\n"
        "JSON output:"
    )
)
_summarize_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following text in 3–5 bullet points.\n\n"
        "**Output must be valid JSON** with a single field `summary` whose value is a string containing the bullets.\n\n"
        "Text to summarize:\n"
        "{text}\n\n"
        "JSON output:"
    )
)
_rewrite_parser = PydanticOutputParser(pydantic_object=RewriteOutput)
_summarize_parser = PydanticOutputParser(pydantic_object=SummarizeOutput)


def load_services(path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    raw = json.load(open(path, encoding="utf-8"))
    entries = raw.get("entries") if isinstance(raw, dict) else raw
    docs = []
    for svc in entries:
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " "],
    )
    return splitter.split_documents(docs)


def build_or_load_index(
    chunks: List[Document],
    embeddings: OpenAIEmbeddings,
    index_dir: Path
) -> FAISS:
    if index_dir.exists():
        return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))
    return vs


def load_questions(path: str = "questions.json") -> List[dict]:
    return json.load(open(path, encoding="utf-8"))


def sample(lst: List, n: int = None, seed: int = 42) -> List:
    r = random.Random(seed)
    return lst if n is None or n >= len(lst) else r.sample(lst, n)


def prompt_int(msg: str, default: Optional[int] = None) -> Optional[int]:
    s = input(msg).strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return prompt_int(msg, default)


def create_rewrite_chain(llm) -> LLMChain:
    return LLMChain(llm=llm, prompt=_rewrite_template, output_parser=_rewrite_parser)


def create_summarize_chain(llm) -> LLMChain:
    return LLMChain(llm=llm, prompt=_summarize_template, output_parser=_summarize_parser)


def get_search_tool() -> DuckDuckGoSearchResults:
    return DuckDuckGoSearchResults()


def throttle(seconds: float = 6.0):
    time.sleep(seconds)


def build_coarse_index(
    dataset_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings: OpenAIEmbeddings,
    index_dir: Path,
) -> FAISS:
    """
    Load or build a ‘coarse’ FAISS index.
    """
    if index_dir.exists():
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    chunks = load_services(dataset_path, chunk_size, chunk_overlap)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))
    return vs


def build_fine_index(
    dataset_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings: OpenAIEmbeddings,
    index_dir: Path,
) -> FAISS:
    """
    Load or build a ‘fine’ FAISS index with arbitrary chunk sizes.
    """
    return build_or_load_index(
        load_services(dataset_path, chunk_size, chunk_overlap),
        embeddings,
        index_dir,
    )


def create_retrieval_qa_chain(
    llm,
    retriever: BaseRetriever,
    chain_type: str = "map_reduce",
) -> RetrievalQA:
    """
    Helper to spin up a RetrievalQA chain.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
    )
