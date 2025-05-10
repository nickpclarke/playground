# -----------------------------------------------------------------------------
# IMPORTS
# Standard library imports
import os       # For interacting with the operating system (e.g., file paths)
import json     # For working with JSON data (encoding and decoding)
import pickle   # For serializing and deserializing Python objects (saving/loading data)
import re       # For regular expression operations (pattern matching in strings)
import html     # For escaping HTML special characters
import datetime # For working with dates and times (e.g., current date)
import subprocess # For running shell commands (e.g., getting Git version info)

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

# Create a two-column layout for the footer
footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.caption("Created with Streamlit • View on [GitHub: nickpclarke/playground](https://github.com/nickpclarke/playground)")

with footer_col2:
    # Generate dynamic version information from Git
    try:
        # Get the latest tag (version) from Git
        git_tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], 
                                          stderr=subprocess.DEVNULL).decode().strip()
        # If no tags exist, this will fail and go to except block
        
        # Get number of commits since tag
        commits_since_tag = subprocess.check_output(
            ["git", "rev-list", f"{git_tag}..HEAD", "--count"], 
            stderr=subprocess.DEVNULL).decode().strip()
        
        # Get current commit hash (short version)
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            stderr=subprocess.DEVNULL).decode().strip()
        
        if int(commits_since_tag) > 0:
            # If there are commits since the last tag, show tag+commits+hash
            APP_VERSION = f"{git_tag}+{commits_since_tag} ({git_commit})"
        else:
            # If we're exactly on a tag, just show the tag
            APP_VERSION = f"{git_tag} ({git_commit})"
            
    except subprocess.SubprocessError:
        try:
            # Fallback: if no tags, use commit count as version
            commit_count = subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"], 
                stderr=subprocess.DEVNULL).decode().strip()
            
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                stderr=subprocess.DEVNULL).decode().strip()
                
            APP_VERSION = f"0.0.{commit_count} ({git_commit})"
        except:
            # Ultimate fallback if Git is not available
            APP_VERSION = "1.0.0-dev"
    
    # Use current date for build date
    BUILD_DATE = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Display version info with right alignment
    st.caption(f"<div style='text-align: right;'>Version {APP_VERSION} • Built {BUILD_DATE}</div>", unsafe_allow_html=True)