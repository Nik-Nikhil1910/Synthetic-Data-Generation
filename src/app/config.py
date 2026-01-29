"""
Application Configuration
=========================

Central configuration for the API.
"""

import os
from pathlib import Path


# =============================================================================
# VERSION
# =============================================================================

VERSION = "1.0.0"
APP_NAME = "Synthetic Data Generator"


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

# Default output directory (can be overridden via environment variable)
OUTPUT_DIR = os.environ.get(
    "SDG_OUTPUT_DIR",
    str(Path(__file__).parent.parent.parent / "output")
)


def get_output_dir() -> str:
    """
    Get the output directory path, creating it if it doesn't exist.
    
    Returns:
        Absolute path to output directory.
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path.resolve())
