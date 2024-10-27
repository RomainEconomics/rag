import weaviate.classes as wvc
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from pydantic import BaseModel, Field, ConfigDict
from langchain.schema.runnable import RunnableSerializable
from langchain_core.language_models import BaseLanguageModel

from rag.chains import (
    AbstractChainBuilder,
    ChainBuilder,
    QAFromContextChain,
    RelevanceChain,
    StructuredOutputFromContextChain,
    ValidateExtraction,
    ValidationChain,
    WeaviateRetrievalChain,
    QAImageChain,
    QAImageStructuredChain,
)
from rag.enums import ChainComponent, ChainType


class LLMConfig(BaseModel):
    """Configuration for different LLMs used in chains"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_llm: BaseLanguageModel
    component_llms: dict[ChainComponent, BaseLanguageModel] = Field(
        default_factory=dict
    )

    def get_llm(self, component: ChainComponent) -> BaseLanguageModel:
        """Get the LLM for a specific component, falling back to default if not specified"""
        return self.component_llms.get(component, self.default_llm)


class ChainFactory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_config: LLMConfig
    vectorstore: WeaviateVectorStore | None = None
    search_type: Literal["similarity", "hybrid"] = "similarity"
    output_schema: type[BaseModel] | None = None
    filters: wvc.query.Filter | None = None
    k: int = 2
    file_path: str | None = None

    def _get_base_retrieval_chain(self) -> AbstractChainBuilder:
        if not self.vectorstore:
            raise ValueError("vectorstore must be provided for retrieval chain")

        return WeaviateRetrievalChain(
            vectorstore=self.vectorstore,
            search_type=self.search_type,
            k=self.k,
            filters=self.filters,
        )

    def _create_basic_qa_chain(self) -> RunnableSerializable:
        retrieval_builder = self._get_base_retrieval_chain()
        qa_builder = QAFromContextChain(
            model=self.llm_config.get_llm(ChainComponent.QA)
        )
        return ChainBuilder(chains=[retrieval_builder, qa_builder]).build()

    def _create_qa_image_chain(self) -> RunnableSerializable:
        if not self.file_path:
            raise ValueError("file_path must be provided for image QA chain")

        retrieval_builder = self._get_base_retrieval_chain()
        qa_image_builder = QAImageChain(
            model=self.llm_config.get_llm(ChainComponent.IMAGE),
            file_path=self.file_path,
        )
        return ChainBuilder(chains=[retrieval_builder, qa_image_builder]).build()

    def _create_qa_image_and_text_chain(self) -> RunnableSerializable:
        retrieval_builder = self._get_base_retrieval_chain()
        qa_builder = QAFromContextChain(
            model=self.llm_config.get_llm(ChainComponent.QA)
        )
        return ChainBuilder(chains=[retrieval_builder, qa_builder]).build()

    def _create_image_structured_output_chain(self) -> RunnableSerializable:
        if not self.file_path:
            raise ValueError(
                "file_path must be provided for image structured output chain"
            )

        if not self.output_schema:
            raise ValueError(
                "output_schema must be provided for structured output chain"
            )

        retrieval_builder = self._get_base_retrieval_chain()
        qa_image_builder = QAImageStructuredChain(
            model=self.llm_config.get_llm(ChainComponent.IMAGE),
            file_path=self.file_path,
            tool=self.output_schema,
        )
        return ChainBuilder(chains=[retrieval_builder, qa_image_builder]).build()

    def _create_structured_output_chain(self) -> RunnableSerializable:
        if not self.output_schema:
            raise ValueError(
                "output_schema must be provided for structured output chain"
            )

        retrieval_builder = self._get_base_retrieval_chain()
        structured_builder = StructuredOutputFromContextChain(
            model=self.llm_config.get_llm(ChainComponent.EXTRACTION),
            tool=self.output_schema,
        )
        return ChainBuilder(chains=[retrieval_builder, structured_builder]).build()

    def _create_relevance_check_chain(self) -> RunnableSerializable:
        if not self.output_schema:
            raise ValueError("output_schema must be provided for relevance check chain")

        retrieval_builder = self._get_base_retrieval_chain()
        relevance_builder = RelevanceChain(
            model=self.llm_config.get_llm(ChainComponent.RELEVANCE)
        )
        structured_builder = StructuredOutputFromContextChain(
            model=self.llm_config.get_llm(ChainComponent.EXTRACTION),
            tool=self.output_schema,
        )
        return ChainBuilder(
            chains=[retrieval_builder, relevance_builder, structured_builder]
        ).build()

    def _create_full_validation_chain(self) -> RunnableSerializable:
        if not self.output_schema:
            raise ValueError("output_schema must be provided for full validation chain")

        retrieval_builder = self._get_base_retrieval_chain()
        relevance_builder = RelevanceChain(
            model=self.llm_config.get_llm(ChainComponent.RELEVANCE)
        )
        structured_builder = StructuredOutputFromContextChain(
            model=self.llm_config.get_llm(ChainComponent.EXTRACTION),
            tool=self.output_schema,
        )
        validation_builder = ValidationChain(
            model=self.llm_config.get_llm(ChainComponent.VALIDATION),
            tool=ValidateExtraction,
        )
        return ChainBuilder(
            chains=[
                retrieval_builder,
                relevance_builder,
                structured_builder,
                validation_builder,
            ]
        ).build()

    def create_chain(self, chain_type: ChainType) -> RunnableSerializable:
        chain_builders = {
            ChainType.BASIC_QA: self._create_basic_qa_chain,
            ChainType.IMAGE_QA: self._create_qa_image_chain,
            ChainType.STRUCTURED_OUTPUT: self._create_structured_output_chain,
            ChainType.STRUCTURED_OUTPUT_IMAGE: self._create_image_structured_output_chain,
            ChainType.RELEVANCE_CHECK: self._create_relevance_check_chain,
            ChainType.FULL_VALIDATION: self._create_full_validation_chain,
        }

        builder = chain_builders.get(chain_type)
        if not builder:
            raise ValueError(f"Unknown chain type: {chain_type}")

        return builder()


class ChainManager:
    def __init__(
        self,
        vectorstore: WeaviateVectorStore | None = None,
        llm_config: LLMConfig | None = None,
    ):
        if llm_config is None:
            llm_config = LLMConfig(
                default_llm=ChatOpenAI(model="gpt-4o-mini"),
            )

        self.factory = ChainFactory(
            vectorstore=vectorstore,
            llm_config=llm_config,
        )
        self.chains: dict[ChainType, RunnableSerializable] = {}

    def get_chain(
        self,
        chain_type: ChainType,
        k: int | None = None,
        output_schema: type[BaseModel] | None = None,
        file_path: str | None = None,
    ) -> RunnableSerializable:
        """Get or create a chain of the specified type"""
        if k is not None:
            self.factory.k = k

        if output_schema is not None:
            self.factory.output_schema = output_schema

        if file_path is not None:
            self.factory.file_path = file_path

        if chain_type not in self.chains:
            self.chains[chain_type] = self.factory.create_chain(chain_type)
        return self.chains[chain_type]

    def run_chain(
        self,
        chain_type: ChainType,
        question: str,
        k: int | None = None,
        output_schema: type[BaseModel] | None = None,
        file_path: str | None = None,
    ):
        """Run a specific chain type with the given question"""
        chain = self.get_chain(
            chain_type=chain_type, k=k, output_schema=output_schema, file_path=file_path
        )
        return chain.invoke({"question": question})
