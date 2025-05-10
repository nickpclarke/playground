"""
Multi-Tool Dashboard Application
A Streamlit application that provides various tools for RSS feed processing,
semantic search, and LLM interactions.
"""
import os
import json
import pickle
import re
import html
import datetime
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
import requests
import xmltodict
import pandas as pd
import numpy as np
import faiss
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px

# Download NLTK data
nltk.download('punkt', quiet=True)

# Import custom modules
from cerebras_client import get_chat_completion, get_text_completion
from openai_client import get_openai_embeddings
from components.rss_tab import render_rss_tab
from components.cerebras_tab import render_cerebras_tab
from components.future_tab import render_future_tab
from config import (
    PAGE_TITLE, PAGE_ICON, LAYOUT, INITIAL_SIDEBAR_STATE,
    TAB_NAMES, DATA_DIR, CACHE_DIR
)

# Initialize session state
if 'rss_data' not in st.session_state:
    st.session_state.rss_data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")

# -----------------------------------------------------------------------------
# TAB CREATION
# -----------------------------------------------------------------------------
tabs = st.tabs(list(TAB_NAMES.values()))

# -----------------------------------------------------------------------------
# RENDER EACH TAB
# -----------------------------------------------------------------------------
with tabs[0]:
    render_rss_tab()

with tabs[1]:
    render_cerebras_tab()

with tabs[2]:
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
        git_tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get number of commits since tag
        commits_since_tag = subprocess.check_output(
            ["git", "rev-list", f"{git_tag}..HEAD", "--count"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get current commit hash (short version)
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
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
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
                
            APP_VERSION = f"0.0.{commit_count} ({git_commit})"
        except:
            # Ultimate fallback if Git is not available
            APP_VERSION = "1.0.0-dev"
    
    # Use current date for build date
    BUILD_DATE = datetime.datetime.now().strftime("%B %d, %Y")
    
    # Display version info with right alignment
    st.caption(
        f"<div style='text-align: right;'>Version {APP_VERSION} • Built {BUILD_DATE}</div>",
        unsafe_allow_html=True
    )