from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import tool
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)


def split_pdf_content(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    all_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(all_documents)
    return text_splitter


def load_embeddings(documents: list[Document]):
    vector_store.add_documents(documents=documents)


@tool(return_direct=True)
def search_pdf(query: str) -> str:
    """Search in visa financial report in query

    Args:
        query (str): search query in document

    Returns:
        str: information about the data in the document
    """

    retriev = vector_store.similarity_search(query=query, k=4)
    return "".join([document.page_content for document in retriev])


def main():
    path = str(Path(__file__).resolve().parents[2] / "documents" / "visa.pdf")

    documents = split_pdf_content(path)
    load_embeddings(documents)

    system_prompt = """You are an expert financial research analyst and institutional reporting assistant.
Your task is to analyze retrieved financial documents (such as SEC filings, 10-Ks, 10-Qs, and earnings transcripts) and synthesize them into comprehensive, accurate, and professional financial reports.

# CORE RULES & GUIDELINES:
# 1. GROUNDING: Base all calculations, metrics, and qualitative statements strictly on the provided context/retrieved documents. Never hallucinate, extrapolate, or assume financial figures not explicitly present in the data.
# 2. CITATIONS: Attribute every key metric, data point, and claim to its specific source document and section (e.g., [10-Q, Q3 2025, Page 14]).
# 3. OBJECTIVITY & TONE: Maintain an objective, formal, and analytical tone appropriate for executive leadership and institutional investors. Avoid sensational or speculative language.
# 4. HANDLING UNCERTAINTY: If the retrieved context lacks sufficient information to answer a specific part of the user's request, explicitly state: "Information regarding [missing topic] was not found in the provided documentation." Do not guess.
# 5. STRUCTURE: Organize the final report logically using clear markdown headers:
#    - Executive Summary
#    - Key Financial Metrics (Revenue, Net Income, Margins, EPS)
#    - Detailed Segment/Fundamental Analysis
#    - Risk Factors & Forward-Looking Outlook
"""

    agent = create_agent(
        "google_genai:gemini-3.6-flash",
        tools=[search_pdf],
        system_prompt=system_prompt,
    )

    EXAMPLE_QUERY = (
        "give me only the Net Revenue from visa in billions dollar q2 as a number"
    )

    result = agent.invoke({"messages": [HumanMessage(content=EXAMPLE_QUERY)]})

    for msg in result.get("messages", []):
        if msg.text:
            print(msg.text)


if __name__ == "__main__":
    main()
