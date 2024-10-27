import os
from typing import Literal
from pydantic import BaseModel
import weaviate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore

from rag.enums import ChainComponent, ChainType
from rag.factory import ChainManager, LLMConfig
from rag.loader import load_pdf
from scripts.weaviate_helper_functions import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    WeaviateCollection,
    create_schema,
    batch_ingest,
)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

weaviate_client = weaviate.connect_to_local(
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)


col = weaviate_client.collections.get(WeaviateCollection.DOCUMENT.value)
# weaviate_client.collections.delete(WeaviateCollection.DOCUMENT.value)

if not col.exists():
    create_schema(weaviate_client)


# First, ingest a pdf
file_path = "Apple_CDP-Climate-Change-Questionnaire_2023.pdf"
docs = load_pdf(file_path)
batch_ingest(weaviate_client, WeaviateCollection.DOCUMENT, docs)

# Document stored in weaviate can be fetched using the following query:
# col.query.fetch_objects()


vectorstore = WeaviateVectorStore(
    weaviate_client,
    WeaviateCollection.DOCUMENT,
    "page_content",
    embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS),
)

filters = None  # wvc.query.Filter.by_property("page").equal(10)


model = ChatOpenAI(model="gpt-4o-mini")


question = "What the GHG scope 1 emission of the company ?"


class GhgEmission(BaseModel):
    scope: Literal["scope1", "scope2_location_based", "scope2_market_based", "scope3"]
    year: int
    value: float
    unit: str


class GhgEmissionData(BaseModel):
    """Ghg emission data for a company for each scope and for different year"""

    data: list[GhgEmission]


# Create LLM configuration with different models for different components
llm_config = LLMConfig(
    # Default model for any component not specifically configured
    default_llm=ChatOpenAI(model="gpt-4o-mini"),
    # Specific models for different components
    component_llms={
        ChainComponent.EXTRACTION: ChatOpenAI(model="gpt-4o-mini"),  # "got-4o"
        ChainComponent.RELEVANCE: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.VALIDATION: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.IMAGE: ChatOpenAI(model="gpt-4o"),
    },
)

# Initialize the chain manager with the LLM configuration
manager = ChainManager(
    vectorstore=vectorstore,
    llm_config=llm_config,
)

# Run a basic QA chain
result = manager.run_chain(
    chain_type=ChainType.BASIC_QA,
    question="What is the GHG scope 1 emission of the company?",
)
result

# Run a structured output chain
result = manager.run_chain(
    chain_type=ChainType.STRUCTURED_OUTPUT,
    question="What is the GHG scope 1 emission of the company?",
    output_schema=GhgEmissionData,
)
result

# Run a chain with relevance check
result = manager.run_chain(
    chain_type=ChainType.RELEVANCE_CHECK,
    question="What is the GHG scope 1 emission of the company?",
    output_schema=GhgEmissionData,
)
result


# Run a chain with full validation
result = manager.run_chain(
    chain_type=ChainType.FULL_VALIDATION,
    question="What is the GHG scope 1 emission of the company?",
    output_schema=GhgEmissionData,
)
result


# Run a chain with image
result = manager.run_chain(
    chain_type=ChainType.IMAGE_QA,
    question="What is the GHG scope 1 emission of the company? If the information is not available, describe what you see in the image(s).",
    file_path="Apple_CDP-Climate-Change-Questionnaire_2023.pdf",
)
result


# Run a chain with image + structured output
result = manager.run_chain(
    chain_type=ChainType.STRUCTURED_OUTPUT_IMAGE,
    question="What is the GHG scope 1 emission of the company? Consider all year available in the context.",
    file_path="Apple_CDP-Climate-Change-Questionnaire_2023.pdf",
    output_schema=GhgEmissionData,
)
result
