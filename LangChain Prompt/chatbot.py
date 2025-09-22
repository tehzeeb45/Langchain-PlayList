from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

# ✅ Set your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# ✅ Use a lightweight, supported model
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",  # ✅ works on free tier
    task="text2text-generation",     # ✅ correct task for T5 models
    temperature=0.7,
    max_new_tokens=256
)

# ✅ System prompt
chat_history = [
    SystemMessage(content='You are a helpful assistant.')
]

# ✅ Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append(HumanMessage(content=user_input))

    # Format full prompt
    prompt = ""
    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            prompt += f"{msg.content}\n"
        elif isinstance(msg, HumanMessage):
            prompt += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            prompt += f"Assistant: {msg.content}\n"

    # Invoke model
    result = llm.invoke(prompt)

    print("AI:", result)
    chat_history.append(AIMessage(content=result))
