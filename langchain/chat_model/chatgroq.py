from langchain_groq import ChatGroq

from dotenv import load_dotenv 

load_dotenv()

# llm = ChatGroq(model= "llama-3.3-70b-versatile")
llm = ChatGroq(model= "openai/gpt-oss-120b")

result = llm.invoke("What is 2+2")

print(result.content)