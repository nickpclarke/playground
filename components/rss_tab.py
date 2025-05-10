import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# Import utility functions
from utils.rss_utils import rss_to_json, extract_feed_items, create_items_dataframe
from utils.text_utils import chunk_text, create_text_chunks
from utils.embedding_utils import (
    generate_embeddings, 
    visualize_embeddings_2d,
    visualize_embeddings_3d,
    perform_semantic_search
)
from openai_client import get_openai_embeddings

def handle_api_key_setup():
    """Check for OpenAI API key and allow user to enter it if missing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and hasattr(st, 'secrets') and "openai" in st.secrets:
        api_key = st.secrets["openai"]["api_key"]
    
    if not api_key:
        with st.expander("OpenAI API Key Setup"):
            st.warning("OpenAI API key not found in environment variables or Streamlit secrets.")
            st.info("You'll need an OpenAI API key for embedding and semantic search features.")
            
            temp_api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if temp_api_key:
                os.environ["OPENAI_API_KEY"] = temp_api_key
                st.success("API key set for this session!")
    
    return api_key is not None

def rss_input_section():
    """Render the RSS feed input section."""
    st.subheader("1. Fetch RSS Feed")

    # Radio button to choose between predefined feeds or a custom URL
    feed_option = st.radio(
        "Choose an RSS feed source:",
        ["Select from defaults", "Enter custom URL"],
        key="rss_feed_option",
        horizontal=True
    )

    # Default RSS feeds dictionary
    default_feeds = {
        "Google News": "https://news.google.com/rss",
        "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
        "CNN News": "http://rss.cnn.com/rss/edition.rss",
        "NASA Breaking News": "https://www.nasa.gov/rss/dyn/breaking_news.rss"
    }

    rss_url = ""
    if feed_option == "Select from defaults":
        selected_feed_key = st.selectbox(
            "Select a default feed:",
            list(default_feeds.keys())
        )
        rss_url = default_feeds[selected_feed_key]
    else:
        rss_url = st.text_input(
            "Enter custom RSS feed URL:",
            "https://news.google.com/rss"
        )

    # Button to trigger the RSS to JSON conversion
    if st.button("Convert to JSON", key="rss_convert_btn"):
        if rss_url:
            with st.spinner("Fetching and converting feed..."):
                json_data, error_message = rss_to_json(rss_url)

                if error_message:
                    st.error(error_message)
                else:
                    st.session_state.rss_json_data = json_data
                    st.success("RSS feed converted and parsed successfully!")
        else:
            st.warning("Please enter a valid RSS feed URL.")

def render_json_view_subtab():
    """Render the JSON View subtab."""
    st.write("Raw JSON structure of the feed:")
    st.json(st.session_state.rss_json_data, expanded=False)

def render_feed_items_subtab():
    """Render the Feed Items & Embed subtab."""
    try:
        # Extract items using utility functions
        items = extract_feed_items(st.session_state.rss_json_data)
        
        if not items:
            st.warning("Could not identify feed item structure. Displaying raw data.")
            st.json(st.session_state.rss_json_data)
            return
        
        # Create DataFrame using utility function
        df = create_items_dataframe(items)
        
        # Display DataFrame
        st.write(f"Found {len(df)} feed items:")
        st.dataframe(df, use_container_width=True)
        
        # Store simplified items for later use in search results
        simplified_items = df.to_dict('records')
        
        # Embedding section
        st.subheader("Semantic Search with Embeddings")
        
        # Add chunk size slider
        chunk_size = st.slider("Chunk size (words)", min_value=20, max_value=200, value=50, step=10,
                              help="Larger chunks provide more context but consume more tokens and may reduce search precision.")
        
        # Create embeddings button
        if st.button("Create Embeddings for Search", key="create_embeddings_btn"):
            with st.spinner("Creating text chunks and embeddings..."):
                # Check for API key
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key and hasattr(st, 'secrets') and "openai" in st.secrets:
                    api_key = st.secrets["openai"]["api_key"]
                
                if not api_key:
                    st.error("OpenAI API key is required for creating embeddings.")
                    st.info("Please set your API key in the 'OpenAI API Key Setup' section.")
                    return
                
                # Create text chunks using utility function
                texts_to_embed, original_item_indices = create_text_chunks(
                    simplified_items, 
                    chunk_size,
                    show_progress=True
                )
                
                if texts_to_embed:
                    # Store texts and indices in session state
                    st.session_state.rss_embedded_texts = texts_to_embed
                    st.session_state.rss_original_item_indices_for_chunks = original_item_indices
                    
                    # Generate embeddings using utility function
                    try:
                        embeddings = generate_embeddings(texts_to_embed, show_progress=True)
                        
                        if embeddings:
                            if len(embeddings) == len(texts_to_embed):
                                embeddings_array = np.array(embeddings).astype('float32')
                                st.session_state.rss_embeddings_array = embeddings_array
                                st.success(f"Created all {len(embeddings)} embeddings successfully!")
                            else:
                                st.warning(f"Created {len(embeddings)} out of {len(texts_to_embed)} embeddings.")
                                # Store what we have if at least half were successful
                                if len(embeddings) >= len(texts_to_embed) / 2:
                                    # Take the first N that succeeded (this is a simplification)
                                    successful_texts = texts_to_embed[:len(embeddings)]
                                    successful_indices = original_item_indices[:len(embeddings)]
                                    
                                    st.session_state.rss_embedded_texts = successful_texts
                                    st.session_state.rss_original_item_indices_for_chunks = successful_indices
                                    
                                    embeddings_array = np.array(embeddings).astype('float32')
                                    st.session_state.rss_embeddings_array = embeddings_array
                                    st.info("Stored the successful embeddings for search.")
                        else:
                            st.error("Failed to generate any embeddings.")
                    except Exception as e_embed:
                        st.error(f"Error during embedding process: {e_embed}")
                else:
                    st.warning("No text to embed found in the feed items.")
        
        # Search and visualization section - only show if embeddings exist
        if 'rss_embeddings_array' in st.session_state and len(st.session_state.rss_embeddings_array) > 0:
            # Create columns for the visualization buttons
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if st.button("Visualize Embeddings (2D)", key="viz_2d_btn"):
                    # Use utility function for 2D visualization
                    visualize_embeddings_2d(
                        st.session_state.rss_embeddings_array, 
                        st.session_state.rss_embedded_texts
                    )
            
            with viz_col2:
                if st.button("Visualize Embeddings (3D)", key="viz_3d_btn"):
                    # Use utility function for 3D visualization
                    visualize_embeddings_3d(
                        st.session_state.rss_embeddings_array, 
                        st.session_state.rss_embedded_texts
                    )
            
            # Search section
            st.subheader("Search Feed Content")

            # Use a form to capture Enter key press
            with st.form(key="search_form"):
                search_query = st.text_input(
                    "Enter search query:", 
                    placeholder="Enter your search query here and press Enter or click Search"
                )
                
                # Add search button within the form
                search_submitted = st.form_submit_button("Search", use_container_width=True)
                
                # Process search when either button is clicked or Enter is pressed
                if search_submitted and search_query:
                    with st.spinner("Searching..."):
                        try:
                            # Use utility function for semantic search
                            perform_semantic_search(
                                search_query,
                                st.session_state.rss_embeddings_array,
                                st.session_state.rss_embedded_texts,
                                st.session_state.rss_original_item_indices_for_chunks,
                                simplified_items
                            )
                        except Exception as e_search:
                            st.error(f"Error during search: {e_search}")
                            st.exception(e_search)
    except Exception as e_display:
        st.error(f"Error displaying or processing feed items: {e_display}")
        st.exception(e_display)

def render_raw_data_subtab():
    """Render the Raw Data Display and Download subtab."""
    st.write("Raw JSON data (as text):")
    raw_json_string = json.dumps(st.session_state.rss_json_data, indent=2)
    st.text_area("JSON Output", raw_json_string, height=300)

    st.download_button(
        label="Download JSON Data",
        data=raw_json_string,
        file_name="rss_feed_data.json",
        mime="application/json",
        key="rss_download_json_btn"
    )

def render_rss_tab():
    """Render the RSS to JSON Converter and Semantic Search tab."""
    st.header("RSS to JSON Converter & Semantic Search")
    
    # Check for OpenAI API key and allow user to enter it if missing
    handle_api_key_setup()

    # --- RSS Input Section (Left Column) and Display Section (Right Column) ---
    col1, col2 = st.columns([1, 2])

    with col1:
        rss_input_section()

    with col2:
        # Check if 'rss_json_data' exists in the session state
        if 'rss_json_data' in st.session_state and st.session_state.rss_json_data:
            st.subheader("2. View & Interact with Feed Data")

            # Create sub-tabs
            rss_subtab1, rss_subtab2, rss_subtab3 = st.tabs([
                "Formatted JSON View", 
                "Feed Items & Embed", 
                "Raw Data & Download"
            ])

            with rss_subtab1:
                render_json_view_subtab()

            with rss_subtab2:
                render_feed_items_subtab()

            with rss_subtab3:
                render_raw_data_subtab()
                
        elif st.button("Clear Cached Feed Data", key="clear_feed_btn_col2"):
            if 'rss_json_data' in st.session_state: del st.session_state.rss_json_data
            if 'rss_embeddings_array' in st.session_state: del st.session_state.rss_embeddings_array
            if 'rss_embedded_texts' in st.session_state: del st.session_state.rss_embedded_texts
            if 'rss_original_item_indices_for_chunks' in st.session_state: del st.session_state.rss_original_item_indices_for_chunks
            st.rerun()
        else:
            st.info("Fetch an RSS feed using the controls on the left to see its content here.")