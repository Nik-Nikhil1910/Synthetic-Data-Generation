"""
FastAPI Application Entry Point
================================

Main application initialization and wiring.
Run with: uvicorn src.app.main:app --reload
"""

# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.app.config import VERSION, APP_NAME
from src.app.exceptions import global_exception_handler


# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title=APP_NAME,
    description="AI-Driven Synthetic Data Generator API",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

app.add_exception_handler(Exception, global_exception_handler)


# =============================================================================
# ROUTES
# =============================================================================

app.include_router(router, tags=["Schema"])


# =============================================================================
# ROOT
# =============================================================================

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": APP_NAME,
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "refine": "POST /schema/refine",
            "generate": "POST /schema/generate"
        }
    }
