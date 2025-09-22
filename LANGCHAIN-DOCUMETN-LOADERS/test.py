from huggingface_hub import InferenceClient

client = InferenceClient(
    model="google/gemma-2-2b-it",
    token="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"  # apna asli write token dal do
)

response = client.text_generation("Hello, how are you?", max_new_tokens=50)
print(response)
