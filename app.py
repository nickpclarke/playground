# -----------------------------------------------------------------------------
# IMPORTS
# Standard library imports
import os       # For interacting with the operating system (e.g., file paths)
import json     # For working with JSON data (encoding and decoding)
import pickle   # For serializing and deserializing Python objects (saving/loading data)
import re       # For regular expression operations (pattern matching in strings)
import html     # For escaping HTML special characters

# Third-party library imports
import streamlit as st  # The main library for building Streamlit web apps
import requests         # For making HTTP requests (e.g., fetching RSS feeds)
import xmltodict        # For converting XML data to Python dictionaries
import pandas as pd     # For data manipulation and analysis (e.g., creating DataFrames)
import numpy as np      # For numerical operations, especially with arrays
import faiss            # For efficient similarity search and clustering of dense vectors

# NLTK (Natural Language Toolkit) for text processing
import nltk
# Download the 'punkt' tokenizer models if not already present.
# 'punkt' is used for sentence tokenization. 'quiet=True' suppresses output.
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize # Specifically import the sentence tokenizer

# Plotting libraries
import matplotlib.pyplot as plt # For creating static, interactive, and animated visualizations
from sklearn.decomposition import PCA # Principal Component Analysis for dimensionality reduction
import plotly.express as px     # For creating interactive plots easily

# Custom local module imports (assuming these files are in the same directory or accessible)
from cerebras_client import get_chat_completion, get_text_completion # Functions for Cerebras API
from openai_client import get_openai_embeddings # Function for OpenAI embeddings API
# -----------------------------------------------------------------------------

# Component imports
from components.rss_tab import render_rss_tab
from components.cerebras_tab import render_cerebras_tab
from components.future_tab import render_future_tab

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Tool App",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="auto"
)

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title("🛠️ Multi-Tool Dashboard")

# -----------------------------------------------------------------------------
# TAB CREATION
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["RSS to JSON & Search", "Cerebras LLM", "Future Project"])

# -----------------------------------------------------------------------------
# RENDER EACH TAB
# -----------------------------------------------------------------------------
with tab1:
    render_rss_tab()

with tab2:
    render_cerebras_tab()

with tab3:
    render_future_tab()

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption("Created with Streamlit • View on [GitHub: nickpclarke/playground](https://github.com/nickpclarke/playground)")
