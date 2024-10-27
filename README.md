# RAG Library

A lightweight Python package for Retrieval Augmented Generation (RAG) using LangChain and Weaviate. This library provides a flexible and modular approach to implementing RAG systems with different types of chains and validation strategies.

When creating RAG application, we often write a lot of boilerplate code to retrieve documents, perform QA, validate the results, and generate structured outputs. This library aims to simplify the process by providing reusable chains and a flexible configuration system to customize the behavior of the RAG system.

At the moment, the library has not yet been published on PyPI (and may never be), but it can still provide a good starting point for building RAG systems with LangChain and Weaviate.

## Features

- Small and reusable chains to perform RAG:

  - Retrieve relevant documents from Weaviate (with filtering)
  - QA from context documents
  - Structured Output from QA
  - Relevance Checking (a model check that each context document is useful to answer the question)
  - Hallucination check (after the QA, check if the answer is consistent with the context)
  - Image-based QA
  - Structured Output with Image Support

- Multiple chain types for different use cases:

  - Basic Question-Answering (Retrieval + QA)
  - Structured Output Generation (Retrieval + QA + Structured Output)
  - Relevance Checking (Retrieval + Relevance Check + QA + Structured Output)
  - Full Validation (Retrieval + Relevance Check + QA + Structured Output + Hallucination Check)
  - Image-based QA (Retrieval + Image QA)
  - Structured Output with Image Support (Retrieval + Image QA + Structured Output)

- Flexible LLM Configuration:

  - Configure different models for different components
  - Default fallback model support

- Vector Store Integration:
  - Seamless integration with Weaviate
  - PDF document ingestion
  - Customizable embedding size and model

## Limitations

At the moment, the library only supports Weaviate as the vectorstore. Future versions may include support for other vector stores, using the LangChain library.

## Future Work

- The ingest strategy is at the moment very basic (simple chunking of the document). Future versions may include more advanced strategies like:
  - parent / child strategy:
    - the llm see the full pages on a pdf, but we perform the similarity/hybrid/bm25 search on the small chunks
  - summarization:
    - perform summarization on each page or chunk, and use the summary for the search / embedding
- Add support for other retrieval strategy (XGBoost model, Colpali)
- Add support for reranking chain (Colbert, Cohere ...)
- Add support for other vector stores (e.g., PG Vector, Faiss)

## Installation

As the library is not yet published on PyPI, you can install it by cloning the GitHub repository and running the following command:

```bash
uv sync
```

This will install the library in your Python environment, leveraging `uv` (best) python manager.

## Quick Start

In a terminal, start the weaviate server:

```bash
docker compose up
```

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from rag.factory import ChainManager, LLMConfig

# Initialize Weaviate client and vector store
vectorstore = WeaviateVectorStore(
    weaviate_client,
    "YourCollection",
    "page_content",
    embedding=OpenAIEmbeddings()
)

# Configure LLM models
llm_config = LLMConfig(
    default_llm=ChatOpenAI(model="gpt-4o-mini"),
    component_llms={
        ChainComponent.EXTRACTION: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.RELEVANCE: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.VALIDATION: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.IMAGE: ChatOpenAI(model="gpt-4o"),
    }
)

# Initialize chain manager
manager = ChainManager(
    vectorstore=vectorstore,
    llm_config=llm_config,
)

# Run a basic QA chain
result = manager.run_chain(
    chain_type=ChainType.BASIC_QA,
    question="Your question here"
)
```

## Chain Types

### Basic QA Chain

Simple question-answering without additional validation or structure:

```python
result = manager.run_chain(
    chain_type=ChainType.BASIC_QA,
    question="What is the GHG scope 1 emission of the company?"
)
```

### Structured Output Chain

Get responses in a structured format using Pydantic models:

```python
class GhgEmission(BaseModel):
    scope: Literal["scope1", "scope2_location_based", "scope2_market_based", "scope3"]
    year: int
    value: float
    unit: str

class GhgEmissionData(BaseModel):
    """Ghg emission data for a company for each scope and for different year"""
    data: list[GhgEmission]


result = manager.run_chain(
    chain_type=ChainType.STRUCTURED_OUTPUT,
    question="What is the GHG scope 1 emission?",
    output_schema=GhgEmission
)
```

### Relevance Check Chain

Includes validation of response relevance:

```python
result = manager.run_chain(
    chain_type=ChainType.RELEVANCE_CHECK,
    question="Your question",
    output_schema=YourSchema
)
```

### Full Validation Chain

Comprehensive validation including relevance and data consistency:

```python
result = manager.run_chain(
    chain_type=ChainType.FULL_VALIDATION,
    question="Your question",
    output_schema=YourSchema
)
```

### Image QA Chain

Question answering with image support:

```python
result = manager.run_chain(
    chain_type=ChainType.IMAGE_QA,
    question="Describe what you see in the image",
    file_path="path/to/your/document.pdf"
)
```

## Configuration

### LLM Configuration

You can configure different models for different components of your RAG system:

```python
llm_config = LLMConfig(
    default_llm=ChatOpenAI(model="gpt-4o-mini"),
    component_llms={
        ChainComponent.EXTRACTION: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.RELEVANCE: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.VALIDATION: ChatOpenAI(model="gpt-4o-mini"),
        ChainComponent.IMAGE: ChatOpenAI(model="gpt-4o"),
    }
)
```

### Vector Store Setup

The library uses Weaviate as the vector store. Here's how to set it up:

```python
weaviate_client = weaviate.connect_to_local(
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

vectorstore = WeaviateVectorStore(
    weaviate_client,
    "YourCollection",
    "page_content",
    embedding=OpenAIEmbeddings(model="text-embedding-ada-002")
)
```

## Document Ingestion

The library doesn't support, directly, PDF ingestion, but this can be done easily.

As an example, you can use the following code to:

- Create a Weaviate collection (equivalent to a table in psql)
- Ingest a PDF document to this new collection

The `create_schema` and `batch_ingest` functions are helper functions that can be found in the `weaviate_helper_functions.py` script.

- `create_schema`: creates the schema for the collection (called `Document`), with the following fields: `page_content`, `filename`, `page`. At ingest time, each chunk will have its own embedding computed based on the `page_content` field.
- `batch_ingest`: ingest a list of documents to the collection.

```python
from rag.loader import load_pdf
from scripts.weaviate_helper_functions import (
    create_schema,
    batch_ingest,
)

col = weaviate_client.collections.get("Document")

if not col.exists():
  create_schema(weaviate_client)

# Load and ingest a PDF
docs = load_pdf("your_document.pdf")
batch_ingest(weaviate_client, "Document", docs)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
