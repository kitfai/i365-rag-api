import time
import json
from pathlib import Path
import httpx
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
# All settings are defined in one place for easy management.
LANGFLOW_URL = "http://localhost:7860/api/v1/run/3d2ddcd1-b7f8-4d7c-9234-a288f8a715f6"
API_KEY = 'sk-FF72rJ3Gtfp17gOyV8KPL5IuiKTokjHbkj-FmUN0Z4A'
# The directory this service will monitor for new PDF files.
WATCH_DIRECTORY = Path(__file__).parent / "pdf_inbox"

# Define the tweaks payload once.
TWEAKS_CONFIG = {
    "ChatInput-4WKag": {
        "should_store_message": False
    }
}

class LangflowFileProcessor(FileSystemEventHandler):
    """
    An event handler that processes new files and sends them to a Langflow endpoint.
    It uses a persistent httpx.Client for efficient connection pooling.
    """
    def __init__(self, client: httpx.Client):
        self.client = client
        print(f"✅ File watcher initialized. Monitoring directory: {WATCH_DIRECTORY}")

    def on_created(self, event):
        """Called when a file or directory is created."""
        if not event.is_directory:
            self.process_file(Path(event.src_path))

    def process_file(self, file_path: Path):
        """Handles the logic of uploading and triggering the Langflow flow."""
        print(f"📄 New file detected: {file_path.name}. Preparing to process...")

        # We use a try-with-resources block to ensure the file is closed.
        try:
            with open(file_path, "rb") as f:
                # Construct the multipart data payload. This is the most robust way.
                multipart_data = {
                    'input_type': (None, 'file'),
                    'output_type': (None, 'chat'),
                    'tweaks': (None, json.dumps(TWEAKS_CONFIG)),
                    'file': (file_path.name, f, 'application/pdf')
                }

                headers = {'x-api-key': API_KEY}

                print(f"🚀 Sending '{file_path.name}' to Langflow...")
                response = self.client.post(LANGFLOW_URL, files=multipart_data, headers=headers, timeout=120.0)
                response.raise_for_status()  # Raise an exception for 4xx/5xx errors

                print(f"✅ Successfully processed '{file_path.name}'.")
                print("Response:", response.json())

        except httpx.HTTPStatusError as exc:
            print(f"❌ HTTP Error for '{file_path.name}': {exc.response.status_code} - {exc.response.text}")
        except httpx.RequestError as exc:
            print(f"❌ Network Error processing '{file_path.name}': {exc}")
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing '{file_path.name}': {e}")

def run_watcher():
    """Sets up and runs the file system watcher."""
    # Ensure the watch directory exists.
    WATCH_DIRECTORY.mkdir(exist_ok=True)

    # Use a context manager for the httpx.Client for clean resource management.
    with httpx.Client() as client:
        event_handler = LangflowFileProcessor(client)
        observer = Observer()
        observer.schedule(event_handler, str(WATCH_DIRECTORY), recursive=False)
        observer.start()
        print("--- Langflow Watcher Service is now running. Press Ctrl+C to stop. ---")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n--- Shutting down watcher service. ---")
        finally:
            observer.stop()
            observer.join()

if __name__ == "__main__":
    run_watcher()