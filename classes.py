import pickle
from pathlib import Path
from typing import Optional
import networkx as nx
from cohere import Client as CohereClient
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import FAISS
from typing import List, Any
from cohere import ClientV2 as CohereClient
from langchain.schema import BaseRetriever, Document


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
        initial_k: int = 8,
        final_k: int = 1,
    ):
        super().__init__(
            base_retriever=base_retriever,
            co=CohereClient(api_key=cohere_api_key),
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
            top_n=self.initial_k,
        )
        return [candidates[r.index] for r in resp.results]


class HypotheticalRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    llm: ChatOpenAI
    embeddings: OpenAIEmbeddings
    k_real: int
    k_hypo: int

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        k_real: int = 5,
        k_hypo: int = 3,
    ):
        super().__init__(
            base_retriever=base_retriever,
            llm=llm,
            embeddings=embeddings,
            k_real=k_real,
            k_hypo=k_hypo,
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        real_docs = self.base_retriever.get_relevant_documents(query)[: self.k_real]
        hypo_snippet = self.llm.predict(f"Write a concise factual paragraph that directly answers: {query}")
        hypo_vec = self.embeddings.embed_documents([hypo_snippet])[0]
        vs = self.base_retriever.vectorstore
        hypo_docs = vs.similarity_search_by_vector(hypo_vec, k=self.k_hypo)
        return real_docs + [d for d in hypo_docs if d not in real_docs]


class RewriteOutput(BaseModel):
    rewritten: str = Field(..., description="Web-search-friendly query")


class SummarizeOutput(BaseModel):
    summary: str = Field(..., description="3–5 bullet summary")


GRAPH_PATH = Path("kg.pkl")


class KnowledgeGraph:
    """
    Builds or loads a chunk‐similarity graph over your document splits.
    """
    def __init__(self, threshold: float = 0.9, path: Path = GRAPH_PATH):
        self.graph = nx.Graph()
        self.threshold = threshold
        self.path = path

    def load_or_build(self, splits: List[Document], embedder: OpenAIEmbeddings):
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.graph = pickle.load(f)
        else:
            self._build(splits, embedder)
            with open(self.path, "wb") as f:
                pickle.dump(self.graph, f)

    def _build(self, splits: List[Document], embedder: OpenAIEmbeddings):
        for i, doc in enumerate(splits):
            self.graph.add_node(i, text=doc.page_content)
        texts = [d.page_content for d in splits]
        embs = embedder.embed_documents(texts)
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(embs)
        n = len(splits)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= self.threshold:
                    self.graph.add_edge(i, j, weight=float(sim[i, j]))


class GraphRAGEngine:
    """
    1) FAISS retrieval
    2) optional Cohere re-ranking
    3) pick highest‐degree node in KG
    4) gather its N-hop neighbors as context
    5) call LLM
    """
    def __init__(
        self,
        vector_store: FAISS,
        kg: KnowledgeGraph,
        llm: ChatOpenAI,
        reranker: Optional[CohereRerankRetriever] = None,
        hops: int = 1,
    ):
        self.vs = vector_store
        self.kg = kg
        self.llm = llm
        self.reranker = reranker
        self.hops = hops

    def query(self, question: str, k: int = 3):
        docs = self.vs.similarity_search(question, k=k)
        if self.reranker:
            docs = self.reranker.get_relevant_documents(question)
        best_node, best_deg = None, -1
        for node, data in self.kg.graph.nodes(data=True):
            if any(d.page_content == data["text"] for d in docs):
                deg = self.kg.graph.degree(node)
                if deg > best_deg:
                    best_deg, best_node = deg, node
        if best_node is None:
            contexts = [docs[0].page_content]
        else:
            nodes = {best_node}
            for _ in range(self.hops):
                nbrs = set().union(*(self.kg.graph.neighbors(n) for n in nodes))
                nodes |= nbrs
            contexts = [self.kg.graph.nodes[n]["text"] for n in nodes]
        merged = "\n---\n".join(contexts)
        prompt = (
            "You are an expert in Ukrainian governmental services.\n"
            f"Context:\n{merged}\n\n"
            f"Question: {question}\n"
            "Answer concisely:"
        )
        answer = self.llm.predict(prompt).strip()
        return answer, contexts


class HybridRetriever(BaseRetriever):
    """Combine a sparse BM25 retriever and a dense FAISS retriever."""
    lex_retriever: BaseRetriever
    sem_retriever: BaseRetriever
    k_lex: int
    k_sem: int

    def __init__(
        self,
        lex_retriever: BaseRetriever,
        sem_retriever: BaseRetriever,
        k_lex: int = 8,
        k_sem: int = 8,
    ):
        super().__init__(
            lex_retriever=lex_retriever,
            sem_retriever=sem_retriever,
            k_lex=k_lex,
            k_sem=k_sem,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs_lex = self.lex_retriever.get_relevant_documents(query)[: self.k_lex]
        docs_sem = self.sem_retriever.get_relevant_documents(query)[: self.k_sem]
        seen = {}
        for d in docs_lex + docs_sem:
            seen[d.metadata.get("id", id(d))] = d
        return list(seen.values())



class PineconeRerankRetriever(BaseRetriever):
    """
    After a first-stage retriever (e.g. CohereRerankRetriever) returns top-k,
    rerank those k with Pinecone’s reranking API and return the top_n.
    """
    retriever: BaseRetriever
    pc_index: Any
    model: str
    top_k: int
    top_n: int

    def __init__(
        self,
        retriever: BaseRetriever,
        pc_index: Any,
        model: str = "bge-reranker-v2-m3",
        top_k: int = 12,
        top_n: int = 1,
    ):
        super().__init__(
            retriever=retriever,
            pc_index=pc_index,
            model=model,
            top_k=top_k,
            top_n=top_n,
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        candidates = self.retriever.get_relevant_documents(query)[: self.top_k]
        texts = [d.page_content for d in candidates]

        resp = self.pc_index.search(
            namespace="",
            query={
                "inputs": {"text": query},
                "top_k": self.top_k,
            },
            rerank={
                "model": self.model,
                "top_n": self.top_n,
                "rank_fields": ["text"],
            },
            fields=["text"],
        )
        return [candidates[hit["index"]] for hit in resp["result"]["hits"]]