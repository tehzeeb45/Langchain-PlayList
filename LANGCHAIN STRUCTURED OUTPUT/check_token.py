import requests

HF_TOKEN = "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"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
r = requests.get("https://huggingface.co/api/whoami", headers=headers)

print(r.status_code, r.text)
