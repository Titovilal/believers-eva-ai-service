from fastapi import FastAPI
from src.api.routes import router
import uvicorn
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

app = FastAPI(title="Believers EVA AI Service")

app.include_router(router)


def main():
    """Main entry point to run the FastAPI application."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
