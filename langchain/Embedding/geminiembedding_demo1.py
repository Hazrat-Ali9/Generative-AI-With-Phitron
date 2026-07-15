from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv 
load_dotenv()

embeddings  = GoogleGenerativeAIEmbeddings(model= "gemini-embedding-2")

vector = embeddings.embed_query("What is the Capital of Bangladesh")

# print(vector)

print(len(vector), vector[:5])