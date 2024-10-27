import enum
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure

from langchain_core.documents import Document


EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024


class WeaviateCollection(str, enum.Enum):
    DOCUMENT = "Document"


def create_schema(client: weaviate.WeaviateClient):
    client.collections.create(
        name=WeaviateCollection.DOCUMENT,
        vectorizer_config=[
            Configure.NamedVectors.text2vec_openai(
                name="page_content_vector",
                source_properties=["page_content"],
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
            )
        ],
        properties=[
            wvc.config.Property(
                name="page_content",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="filename",
                data_type=wvc.config.DataType.TEXT,
            ),
            wvc.config.Property(
                name="page",
                data_type=wvc.config.DataType.NUMBER,
            ),
        ],
    )


def batch_ingest(
    client: weaviate.WeaviateClient, col: WeaviateCollection, data: list[Document]
):
    formatted_data = _format_documents(data)
    with client.batch.dynamic() as batch:
        for data_row in formatted_data:
            batch.add_object(
                properties=data_row,
                collection=col,
            )

    print("Failed objects:")
    print(client.batch.failed_objects)


def _format_documents(docs: list[Document]) -> list[dict]:
    return [
        {
            "page_content": doc.page_content,
            "page": doc.metadata["page"] + 1,
            "filename": doc.metadata["source"],
        }
        for doc in docs
    ]
