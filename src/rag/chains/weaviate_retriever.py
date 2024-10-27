import weaviate.classes as wvc
from typing import Literal
from operator import itemgetter


from langchain_weaviate import WeaviateVectorStore
from langchain_core.runnables import RunnablePassthrough

from rag.chains.builder import AbstractChainBuilder


##################################################
# Retrieval Chain With Weaviate
##################################################


class WeaviateRetrievalChain(AbstractChainBuilder):
    vectorstore: WeaviateVectorStore
    filters: wvc.query.Filter | None = None
    search_type: Literal["similarity", "hybrid"]
    k: int = 2

    def build(self):
        retriever = self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.k, "filters": self.filters},
        )
        retrieval_chain = (
            RunnablePassthrough.assign(question=itemgetter("question"))
            | RunnablePassthrough.assign(
                source_documents=lambda _: itemgetter("question") | retriever
            )
            | RunnablePassthrough.assign(
                context=lambda inputs: self._format_docs(inputs["source_documents"])
            )
        )
        return retrieval_chain

    def _format_docs(self, docs):
        s = ""
        for doc in docs:
            s += f"filename: {doc.metadata['filename']}; page: {doc.metadata['page']}\n{doc.page_content}\n\n"
        return s
