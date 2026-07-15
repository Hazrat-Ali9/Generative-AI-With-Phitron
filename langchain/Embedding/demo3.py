from langchain_huggingface import HuggingFaceEmbeddings 

embedding = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

docuemnts = [
    "I love AI",
    "I love Machine Leanring", 
    "I love Deep Learning"
]

vector = embedding.embed_documents(docuemnts)

# print(len(vector))
print(vector[0])