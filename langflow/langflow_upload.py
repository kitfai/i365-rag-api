import requests
import base64
import json

# === Configuration ===
LANGFLOW_HOST = "http://localhost:7860"  # Adjust if hosted remotely
FLOW_ID = "3d2ddcd1-b7f8-4d7c-9234-a288f8a715f6"  # Replace with actual flow UUID
FILE_PATH = r"D:\Codes\pdf-rag\pdf\Progress billing 8.pdf"  # File to upload

# === Read and Encode File ===
with open(FILE_PATH, "rb") as f:
    file_bytes = f.read()
    file_base64 = base64.b64encode(file_bytes).decode("utf-8")

# === Construct File Message ===
file_message = {
    "type": "content_block",  # or "content_block" depending on node
    "data": {
        "name": FILE_PATH,
        "mime_type": "application/pdf",
        "content": file_base64
    }
}

# === Send POST to Trigger Flow ===
response = requests.post(
    f"{LANGFLOW_HOST}/api/v1/run/{FLOW_ID}",
    json={"inputs": {"input": file_message}}  # "input" should match the entry node's input key
)

# === Handle Output ===
if response.status_code == 200:
    print("Flow executed successfully.")
    print(json.dumps(response.json(), indent=2))
else:
    print("Failed to run flow:", response.status_code,response.text)