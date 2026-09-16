import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import tool
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

PSQL_URL = os.getenv("PSQL_URL")
if not PSQL_URL:
    sys.exit("Error: PSQL_URL is missing or empty in .env")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="financial_reports",
    connection=PSQL_URL,
)


class FinancialMetrics(BaseModel):
    company: str = Field(description="Company name, e.g., Visa")
    metric_name: str = Field(
        description="Name of the financial metric (e.g., Net Revenue)"
    )
    period: str = Field(description="Fiscal period or quarter (e.g., Q2)")
    value: float = Field(description="Exact numerical value without symbols or text")
    unit: str = Field(description="Unit of measurement (e.g., billions USD)")


def _process_single_pdf(file_path: Path) -> list[Document]:
    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        ).split_documents(documents)
    except Exception as e:
        print(f"Failed to process {file_path.name}: {e}")
        return []


def split_pdf_content(files: list[Path], max_workers: int) -> list[Document]:
    all_documents: list[Document] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_process_single_pdf, file_path): file_path
            for file_path in files
        }

        for future in as_completed(future_to_file):
            docs = future.result()
            all_documents.extend(docs)

    return all_documents


def load_embeddings(documents: list[Document]):
    vector_store.add_documents(documents=documents)


@tool
def search_pdf(query: str) -> str:
    """Search in vectore store about financial report from query

    Args:
        query (str): search query in document

    Returns:
        str: information about the data in the document
    """

    retriev: list[Document] = vector_store.similarity_search(query=query)
    return "\n\n".join([document.page_content for document in retriev])


def main():
    dir: Path = (Path.cwd() / "documents").resolve()
    max_workers: int = os.cpu_count() or 1

    pdf_files: list[Path] = [file for file in dir.glob("*.pdf") if file.is_file()]

    documents: list[Document] = split_pdf_content(pdf_files, max_workers)
    load_embeddings(documents)

    system_prompt = (
        "You are an expert financial research analyst."
        "Use the search_pdf tool to retrieve accurate financial data."
        "Extract the requested metrics precisely based on the retrieved documents."
    )

    agent = create_agent(
        "google_genai:gemini-3.6-flash",
        tools=[search_pdf],
        system_prompt=system_prompt,
        response_format=FinancialMetrics,
    )

    EXAMPLE_QUERY = "Give me the Net Revenue from Visa in billions of dollars for Q1."

    result = agent.invoke({"messages": [HumanMessage(content=EXAMPLE_QUERY)]})

    structured_data: FinancialMetrics = result["structured_response"]

    print(flush=True)
    pprint(structured_data)


if __name__ == "__main__":
    main()
