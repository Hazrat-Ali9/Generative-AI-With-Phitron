from langchain_huggingface import HuggingFaceEmbeddings 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "I love programming in Python, it's my favorite language",
    "JavaScript is great for building web applications",
    "Eating pizza with friends on a Friday night",
    "Python is awesome for data science and machine learning",
    "I enjoy hiking in the mountains during summer"
]

query = "I like python"

doc_vector = embedding.embed_documents(documents)
query_vector = embedding.embed_query(query)

scores = cosine_similarity([query_vector],doc_vector)[0]

print(scores)

index = np.argmax(scores)
score = scores[index]
# print(scores)


print("query",query)
print("Most similar",documents[index])
print("Similarity Score",score)
