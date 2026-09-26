"""
pytest configuration for ReviewPulse.
Adds the project root directory to sys.path so imports work correctly.
"""

import sys
from pathlib import Path

# project root directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
