import os
import re

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# LLM
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
)

model = ChatHuggingFace(llm=llm)


# ============================================================
# DUCKDUCKGO SEARCH
# ============================================================

search = DuckDuckGoSearchResults()


@tool
def ddg_search(query: str) -> str:
    """
    Search the web for official technical documentation.
    Return search results containing documentation URLs.
    """

    return search.run(query)


# ============================================================
# SOURCE DISCOVERY AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=[ddg_search],
    system_prompt="""
You are a documentation source discovery agent.

Your ONLY job is to find the most relevant CURRENT
official documentation URL for the user's technical question.

Rules:

1. Search the web using the search tool.
2. Prefer official documentation.
3. Prefer current documentation.
4. Do not use deprecated or archived URLs.
5. Do not answer the technical question.
6. Return exactly ONE URL.
7. Return only the URL.
8. Do not use Markdown.
9. Do not explain your choice.

Example:

User:
How do I call agents in LangChain?

Output:
https://docs.langchain.com/oss/python/langchain/agents
""",
)


# ============================================================
# EXTRACT URL
# ============================================================

def extract_url(text: str):

    urls = re.findall(
        r"https?://[^\s<>\"']+",
        text,
    )

    if not urls:
        return None

    return urls[0].rstrip(".,);]")


# ============================================================
# FIND DOCUMENTATION URL
# ============================================================

def find_documentation_url(question: str):

    print("\n" + "=" * 60)
    print("SOURCE DISCOVERY")
    print("=" * 60)

    print(f"Question: {question}")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    agent_output = result["messages"][-1].content

    print("\nAgent output:")
    print(agent_output)

    url = extract_url(agent_output)

    if not url:

        print("\nNo URL found.")

        return None

    print("\nExtracted URL:")
    print(url)

    return url


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk a documentation question: "
    )

    url = find_documentation_url(question)

    print("\nFinal URL:")
    print(url)