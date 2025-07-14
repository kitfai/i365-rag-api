# D:/Codes/rhb-poc/webapi/app/main.py

from fastapi import FastAPI
from .routers import ragllm

# 1. Create the main application that will contain all your logic
#    Let's call it `api_app` to be clear.
api_app = FastAPI(
    title="I365 API",
    description="API for I365.",
    version="1.0.0",
)

# 2. Include your routers on this `api_app`
api_app.include_router(ragllm.router)

# 3. Define your root endpoint on this `api_app` as well
@api_app.get("/", tags=["API Root"])
def read_api_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to I365 API v1.0"}

# 4. Create the final, top-level app. This is what Uvicorn will run.
app = FastAPI()

# 5. Mount your entire `api_app` under the desired prefix.
app.mount("/api/v1.0", api_app)

# You can add a simple health check to the absolute root if you want
@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok"}