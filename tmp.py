from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
    temperature=0.2,
)

resp = llm.invoke("Reply with exactly: OK_OPENROUTER")
print(resp.content)
