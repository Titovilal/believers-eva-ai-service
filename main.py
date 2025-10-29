from fastapi import FastAPI
from src.api.routes import router
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Believers EVA AI Service")

app.include_router(router)


def main():
    """Main entry point to run the FastAPI application."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
