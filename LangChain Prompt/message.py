from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation",  # required!
    temperature=0.7,
    max_new_tokens=256
)
message =[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Hi')

]
result = llm.invoke(message)
message.append(AIMessage(content=result.content))
print(message)