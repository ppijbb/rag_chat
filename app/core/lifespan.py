from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.service.medical_inquiry import MedicalInquiryService
from app.core.langchain_module.llm import DDG_LLM

@asynccontextmanager
async def service_lifecycle(app: FastAPI):
    # Initialize services
    app.state.global_llm = DDG_LLM()
    
    # Load models and resources
    try:
        # TODO: Initialize any ML models, database connections, etc.
        pass
    except Exception as e:
        print(f"Error during startup: {e}")
        raise e

    print("Application startup complete")

    try:
        yield
    finally:
        # Cleanup resources
        print("Shutting down application...")
        # TODO: Close any connections, cleanup resources
        print("Application shutdown complete")
