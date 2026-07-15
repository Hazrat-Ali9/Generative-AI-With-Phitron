from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv 
load_dotenv()

embeddings  = GoogleGenerativeAIEmbeddings(model= "gemini-embedding-2")
vectors = embeddings.embed_documents(
    [
        "Today is Monday",
        "Today is Tuesday",
        "Today is April Fools day",
    ]
)

print(len(vectors), len(vectors[0]))