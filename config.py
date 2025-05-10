"""
Configuration settings for the Multi-Tool Dashboard application.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# RSS Configuration
RSS_INDEX_FILE = DATA_DIR / "rss.index"
RSS_META_FILE = DATA_DIR / "rss_meta.pkl"
RSS_TITLES_INDEX_FILE = DATA_DIR / "rss_titles.index"
RSS_TITLES_META_FILE = DATA_DIR / "rss_titles_meta.pkl"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536

# Cache Configuration
CACHE_TTL = 3600  # 1 hour in seconds

# UI Configuration
PAGE_TITLE = "Multi-Tool Dashboard"
PAGE_ICON = "🛠️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "auto"

# Tab Names
TAB_NAMES = {
    "rss": "RSS to JSON & Search",
    "cerebras": "Cerebras LLM",
    "future": "Future Project"
} 