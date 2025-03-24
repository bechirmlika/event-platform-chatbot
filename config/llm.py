from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=settings.GEMINI_API_KEY)

llm = get_llm()
print("✅ Gemini 2.0 LLM loaded successfully!")
