from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    huggingfacehub_api_token="your_token_here"
)

response = llm.invoke("Hello there!")
print(response)
