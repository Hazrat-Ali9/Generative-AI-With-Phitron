from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    task = "conversational", 
    max_new_tokens =100,  
)

model = ChatHuggingFace(llm=llm)


result = model.invoke("Write me a poem on AI")

print(result.content)