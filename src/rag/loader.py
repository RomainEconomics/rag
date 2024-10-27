from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import CharacterTextSplitter


def load_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 0):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)

    return docs
