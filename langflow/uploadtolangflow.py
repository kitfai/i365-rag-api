import httpx
import requests
import json
def upload_file_to_langflow(file_path, langflow_url, api_key):
    """
    Uploads a file to Langflow using the /api/v2/files endpoint.

    Args:
        file_path (str): The path to the file to be uploaded.
        langflow_url (str): The base URL of your Langflow instance.
        api_key (str): Your Langflow API key.

    Returns:
        dict: The JSON response from Langflow, typically containing file information.
    """
    files = {'file': open(file_path, 'rb')}
    headers = {'x-api-key': 'sk-FF72rJ3Gtfp17gOyV8KPL5IuiKTokjHbkj-FmUN0Z4A'}
    #url = f"{langflow_url}/api/v2/files"
    url = f"{langflow_url}/api/v1/files/upload/3d2ddcd1-b7f8-4d7c-9234-a288f8a715f6"

    try:
        response = httpx.post(url, files=files, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        print( response.json())
        langflow_url_ = "http://localhost:7860/api/v1/run/3d2ddcd1-b7f8-4d7c-9234-a288f8a715f6"
        response = requests.post(langflow_url_, headers=headers)

        print("Status Code:", response.status_code)
        # Use a try-except block for JSON decoding in case the server returns non-JSON error
        try:
            print("Response:", response.json())
        except json.JSONDecodeError:
            print("Response (not valid JSON):", response.text)
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        return None
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}: {exc.response.text}")
        return None

# Example usage:
langflow_server_address = "http://localhost:7860"
langflow_api_key = "your_api_key_here"
file_to_upload = r"D:\Codes\pdf-rag\pdf\Progress billing 8.pdf"
uploaded_file_info = upload_file_to_langflow(file_to_upload, langflow_server_address, langflow_api_key)
if uploaded_file_info:
    print(f"File uploaded successfully: {uploaded_file_info}")