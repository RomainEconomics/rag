from operator import itemgetter

from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.chains.builder import AbstractChainBuilder


##################################################
# Relevance Chain
##################################################


class DocumentRelevance(BaseModel):
    relevant: bool = Field(
        description="Whether the document is relevant to the question"
    )
    explanation: str = Field(
        description="Brief explanation of why the document is relevant or not"
    )


class RelevanceCheck(BaseModel):
    documents: dict[str, DocumentRelevance] = Field(
        description="Relevance check for each document"
    )


class RelevanceChain(AbstractChainBuilder):
    model: BaseLanguageModel
    prompt: str | None = None

    def build(self):
        if not self.prompt:
            self.prompt = self._default_template
        prompt = ChatPromptTemplate.from_template(self.prompt)

        _relevance_chain = (
            {
                "question": itemgetter("question"),
                "doc": itemgetter("doc"),
            }
            | RunnablePassthrough.assign(
                page_content=lambda inputs: inputs["doc"].page_content
            )
            | RunnablePassthrough.assign(
                response=lambda _: prompt
                | self.model.with_structured_output(DocumentRelevance)
            )
        )

        def _relevance_chain_by_batch(inputs):
            return _relevance_chain.batch(
                [
                    {
                        "question": inputs["question"],
                        "doc": doc,
                    }
                    for doc in inputs["source_documents"]
                ]
            )

        def _filter_relevant_docs(inputs):
            docs = inputs["relevance_response"]
            return [doc["doc"] for doc in docs if doc["response"].relevant]

        return (
            RunnablePassthrough.assign(
                relevance_response=RunnableLambda(_relevance_chain_by_batch)
            )
            | RunnablePassthrough.assign(  # we overwrite source_docs to not modify other chains
                raw_source_documents=lambda inputs: inputs["source_documents"]
            )
            | RunnablePassthrough.assign(source_documents=_filter_relevant_docs)
            | RunnablePassthrough.assign(
                context=lambda inputs: self._format_docs(inputs["source_documents"])
            )
        )

    @property
    def _default_template(self):
        return """
                Determine if the following document is relevant to answering the given question.
                Return your decision as a structured output.

                Question: {question}

                Document Content: {page_content}

                Relevant or not:
                """

    def _format_docs(self, docs):
        s = ""
        for doc in docs:
            s += f"filename: {doc.metadata['filename']}; page: {doc.metadata['page']}\n{doc.page_content}\n\n"
        return s
