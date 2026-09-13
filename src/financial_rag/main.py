from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()


def main():
    model = init_chat_model(
        "google_genai:gemini-3.6-flash",
        max_retries=10,
        timeout=120,
    )

    system_msg = SystemMessage("""
You are an elite financial AI assistant and quantitative analyst. Your primary goal is to extract, analyze, and synthesize data from provided financial documents with absolute precision.

### CRITICAL CONSTRAINTS & RULES:
1. **Strict Grounding:** Answer ONLY using the facts explicitly provided in the retrieved context chunks. Do not extrapolate, assume, or use parametric/prior knowledge.
2. **Handling Insufficient Data:** If the retrieved context does not contain the exact answer or necessary data points, state clearly: "I cannot find sufficient information in the provided documents to answer this question." Never guess or hallucinate numbers.
3. **Numerical & Table Accuracy:** Preserve exact units and fiscal periods.
4. **Mandatory Citations:** Every single factual claim must be followed by a citation.
5. **Tone & Style:** Clinical, professional, objective, and concise.
""")

    user_question = """
    Question: Give me the latest quarter report from visa.
    """

    messages = [system_msg, HumanMessage(content=user_question)]

    response = model.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    main()
