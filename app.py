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

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# These functions perform specific tasks and help keep the main app logic cleaner.
# -----------------------------------------------------------------------------

def rss_to_json(url: str) -> tuple[dict | None, str | None]:
    """
    Fetches an RSS feed from the given URL and converts its XML content to a JSON-like Python dictionary.

    Args:
        url (str): The URL of the RSS feed.

    Returns:
        tuple[dict | None, str | None]: A tuple containing the parsed data as a dictionary
                                         and None if successful, or (None, error_message_string)
                                         if an error occurred.
    """
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url, timeout=10) # Added a timeout for robustness
        # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()
        # Get the raw XML content from the response
        xml_content = response.content
        # Parse the XML content into a Python dictionary using xmltodict
        # The 'dict_constructor=dict' ensures it uses standard dicts.
        parsed_data = xmltodict.parse(xml_content, dict_constructor=dict)
        return parsed_data, None
    except requests.exceptions.Timeout:
        return None, f"Error: The request to {url} timed out."
    except requests.exceptions.RequestException as e:
        # Handle network-related errors (e.g., DNS failure, connection error)
        return None, f"Error fetching RSS feed: {e}"
    except xmltodict.ParsingError as e:
        # Handle errors that occur if the XML is malformed
        return None, f"Error parsing XML: {e}"
    except Exception as e:
        # Catch any other unexpected errors
        return None, f"An unexpected error occurred: {e}"

def chunk_text(text: str, *, max_tokens: int = 50) -> list[str]:
    """
    Splits a given text into smaller chunks, each not exceeding `max_tokens`.
    This version uses a simple word-based splitting strategy.
    It's useful for preparing text for language models that have token limits.

    Args:
        text (str): The input text to be chunked.
        max_tokens (int, optional): The maximum number of words (approximating tokens)
                                    allowed in each chunk. Defaults to 50.

    Returns:
        list[str]: A list of text chunks.
    """
    # Split the text into individual words using whitespace as a delimiter
    words = text.split()

    # If there are no words, return an empty list
    if not words:
        return []

    chunks = []          # List to store the resulting text chunks
    current_chunk = []   # List to build the current chunk of words
    current_length = 0   # Tracks the number of words in the current_chunk

    # Iterate over each word in the input text
    for word in words:
        # If adding the current word (plus a space, approximated by +1)
        # would exceed max_tokens, finalize the current_chunk.
        # (Note: A more precise token counter would use a tokenizer library)
        if current_length + len(word.split()) > max_tokens and current_chunk: # Ensure current_chunk is not empty
            chunks.append(' '.join(current_chunk)) # Join words in current_chunk to form a string
            current_chunk = [word]                 # Start a new chunk with the current word
            current_length = len(word.split())     # Reset length for the new chunk
        else:
            # Otherwise, add the word to the current_chunk
            current_chunk.append(word)
            current_length += len(word.split())

    # After the loop, if there are any words left in current_chunk, add them as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# This should be the first Streamlit command in your app.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Tool App",  # Title that appears in the browser tab
    page_icon="🛠️",              # Icon for the browser tab (can be an emoji or URL)
    layout="wide",               # Use the full page width for content
    initial_sidebar_state="auto" # How the sidebar (if any) should be initially displayed
)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title("🛠️ Multi-Tool Dashboard") # Display the main title of the application
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# TAB CREATION
# Organize different parts of the application into tabs for better navigation.
# -----------------------------------------------------------------------------
# `st.tabs` returns a list of Tab objects, one for each label provided.
tab1, tab2, tab3 = st.tabs(["RSS to JSON & Search", "Cerebras LLM", "Future Project"])
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# TAB 1: RSS to JSON Converter and Semantic Search
# -----------------------------------------------------------------------------
with tab1: # This block defines the content within the first tab
    st.header("RSS to JSON Converter & Semantic Search") # Header for this tab

    # --- RSS Input Section (Left Column) ---
    # `st.columns` creates a layout with multiple columns.
    # The argument [1, 2] means the first column takes 1 part of the width,
    # and the second column takes 2 parts (i.e., col2 is twice as wide as col1).
    col1, col2 = st.columns([1, 2])

    with col1: # Content for the first (left) column
        st.subheader("1. Fetch RSS Feed") # Subheader for this section

        # Radio button to choose between predefined feeds or a custom URL
        feed_option = st.radio(
            "Choose an RSS feed source:", # Label for the radio button
            ["Select from defaults", "Enter custom URL"], # Options
            key="rss_feed_option", # Unique key for this widget (important for state)
            horizontal=True # Display options horizontally
        )

        # Default RSS feeds dictionary
        default_feeds = {
            "Google News": "https://news.google.com/rss",
            "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
            "CNN News": "http://rss.cnn.com/rss/edition.rss",
            "NASA Breaking News": "https://www.nasa.gov/rss/dyn/breaking_news.rss"
        }

        rss_url = "" # Initialize rss_url
        if feed_option == "Select from defaults":
            # Dropdown to select from the list of default feeds
            selected_feed_key = st.selectbox(
                "Select a default feed:", # Label for the selectbox
                list(default_feeds.keys()) # Options are the keys of the default_feeds dict
            )
            rss_url = default_feeds[selected_feed_key] # Get the URL for the selected feed
        else:
            # Text input for entering a custom RSS feed URL
            rss_url = st.text_input(
                "Enter custom RSS feed URL:", # Label for the text input
                "https://news.google.com/rss" # Default value in the input box
            )

        # Button to trigger the RSS to JSON conversion
        if st.button("Convert to JSON", key="rss_convert_btn"):
            if rss_url: # Proceed only if an RSS URL is provided
                # `st.spinner` shows a loading message while the enclosed code runs
                with st.spinner("Fetching and converting feed..."):
                    # Call the helper function to get and parse the RSS feed
                    json_data, error_message = rss_to_json(rss_url)

                    if error_message:
                        # If an error occurred, display it
                        st.error(error_message)
                    else:
                        # If successful, store the JSON data in Streamlit's session state.
                        # `st.session_state` allows preserving data across reruns of the script.
                        st.session_state.rss_json_data = json_data
                        st.success("RSS feed converted and parsed successfully!")
            else:
                # If no URL is provided, show a warning
                st.warning("Please enter a valid RSS feed URL.")

    # --- RSS Output and Interaction Section (Right Column) ---
    with col2: # Content for the second (right) column
        # Check if 'rss_json_data' exists in the session state (i.e., if a feed has been converted)
        if 'rss_json_data' in st.session_state and st.session_state.rss_json_data:
            st.subheader("2. View & Interact with Feed Data") # Subheader for this section

            # Create sub-tabs within this column for different views/actions
            rss_subtab1, rss_subtab2, rss_subtab3 = st.tabs(["Formatted JSON View", "Feed Items & Embed", "Raw Data & Download"])

            # --- Sub-Tab 1: Formatted JSON View ---
            with rss_subtab1:
                st.write("Raw JSON structure of the feed:")
                # `st.json` displays a Python dictionary (or JSON string) in an interactive, collapsible format
                st.json(st.session_state.rss_json_data, expanded=False) # Initially collapsed

            # --- Sub-Tab 2: Feed Items Table, Embedding, and Search ---
            with rss_subtab2:
                try:
                    items = [] # Initialize list to store feed items
                    # Attempt to extract items from common RSS/Atom feed structures
                    # RSS 2.0 structure
                    if 'rss' in st.session_state.rss_json_data and \
                       'channel' in st.session_state.rss_json_data['rss'] and \
                       'item' in st.session_state.rss_json_data['rss']['channel']:
                        items = st.session_state.rss_json_data['rss']['channel']['item']
                    # Atom feed structure
                    elif 'feed' in st.session_state.rss_json_data and \
                         'entry' in st.session_state.rss_json_data['feed']:
                        items = st.session_state.rss_json_data['feed']['entry']
                    else:
                        st.warning("Could not automatically identify feed item structure. Displaying raw data if possible.")

                    # Ensure 'items' is a list (some feeds might return a single item as a dict)
                    if items and not isinstance(items, list):
                        items = [items]

                    if items:
                        st.write("Extracted Feed Items:")
                        # Create a Pandas DataFrame from the extracted items for tabular display
                        # We're interested in 'title' and 'link'.
                        # `.get(key, default_value)` is used to safely access dictionary keys.
                        df_items = pd.DataFrame([{
                            'Title': item.get('title', 'No Title Provided'),
                            'Link': item.get('link', item.get('guid', 'No Link Provided')) # 'guid' can sometimes be a link
                        } for item in items])

                        # Display the DataFrame as an interactive table
                        st.dataframe(df_items, use_container_width=True)

                        st.divider() # Visual separator
                        st.write("Embed Titles for Semantic Search & Visualization:")

                        # Layout for chunk size input and ingest button
                        ingest_col1, ingest_col2 = st.columns([1, 1])
                        with ingest_col1:
                            # Number input for user to define chunk size for text embedding
                            chunk_size_for_embedding = st.number_input(
                                "Chunk size (words per chunk):",
                                min_value=5,    # Minimum allowed chunk size
                                max_value=200,  # Maximum allowed chunk size
                                value=50,       # Default chunk size
                                step=5,         # Increment/decrement step
                                help="Maximum words per text chunk before embedding. Shorter chunks can be better for semantic meaning of headlines."
                            )
                        with ingest_col2:
                            # Button to trigger ingestion and embedding of feed item titles
                            if st.button("Ingest Titles into Vector DB", key="ingest_btn"):
                                if not df_items.empty:
                                    with st.spinner("Embedding titles and building index..."):
                                        # Prepare texts for embedding (titles, potentially chunked)
                                        texts_to_embed = []
                                        original_item_indices = [] # To map chunks back to original items for metadata

                                        for index, row in df_items.iterrows():
                                            title_text = str(row['Title']) # Ensure title is a string
                                            # Chunk the title text using the helper function
                                            title_chunks = chunk_text(title_text, max_tokens=chunk_size_for_embedding)
                                            texts_to_embed.extend(title_chunks)
                                            # Keep track of which original item each chunk belongs to
                                            original_item_indices.extend([index] * len(title_chunks))


                                        if not texts_to_embed:
                                            st.warning("No text found in titles to embed.")
                                        else:
                                            # Get embeddings for the texts using OpenAI
                                            embeddings_array = np.array(get_openai_embeddings(texts_to_embed)).astype("float32")

                                            # Build a FAISS index for efficient similarity search
                                            embedding_dimension = embeddings_array.shape[1] # Dimension of the embeddings
                                            faiss_index = faiss.IndexFlatL2(embedding_dimension) # Using L2 distance
                                            faiss_index.add(embeddings_array) # Add embeddings to the index

                                            # Persist the FAISS index and metadata to disk
                                            faiss.write_index(faiss_index, "rss_titles.index")

                                            # Prepare metadata: we need the original titles and links,
                                            # but our `texts_to_embed` might contain chunks.
                                            # We'll store the original items' data.
                                            # The search will return indices into the `embeddings_array` (and thus `texts_to_embed`).
                                            # We'll need to map these back to the original items.
                                            # For simplicity in this version, the metadata will store the original items.
                                            # A more advanced setup might store chunk-specific metadata or map chunks to items.
                                            # Here, we'll store the original DataFrame's records.
                                            # The search results will point to chunks, but we'll display the original item.
                                            metadata_for_persistence = df_items[["Title", "Link"]].to_dict("records")
                                            with open("rss_titles_meta.pkl", "wb") as f_meta:
                                                pickle.dump(metadata_for_persistence, f_meta)

                                            # Store embeddings and original texts (or chunks) in session state for visualization
                                            st.session_state.rss_embeddings_array = embeddings_array
                                            st.session_state.rss_embedded_texts = texts_to_embed # These are the chunks
                                            st.session_state.rss_original_item_indices_for_chunks = original_item_indices # Map chunks to original items

                                            st.success(f"Successfully ingested and embedded {len(texts_to_embed)} text chunks into FAISS index!")
                                else:
                                    st.warning("No items in DataFrame to ingest.")

                        # --- Visualization Buttons ---
                        # These buttons appear only if embeddings are present in the session state
                        if "rss_embeddings_array" in st.session_state:
                            st.write("Visualize Embedded Titles (PCA Projection):")
                            viz_col1, viz_col2 = st.columns(2)
                            with viz_col1:
                                if st.button("2D Scatter Plot", key="viz_2d_btn"):
                                    embeddings_to_visualize = st.session_state.rss_embeddings_array
                                    texts_for_labels = st.session_state.rss_embedded_texts # Labels are the chunks

                                    # Reduce dimensionality to 2D using PCA
                                    pca_2d = PCA(n_components=2)
                                    coords_2d = pca_2d.fit_transform(embeddings_to_visualize)

                                    # Create a Matplotlib scatter plot
                                    fig, ax = plt.subplots(figsize=(8, 8)) # Adjusted size
                                    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.7, s=15) # Adjusted alpha and size
                                    
                                    # Add small text labels for each point (or a subset for performance)
                                    # You can adjust the number of labels (e.g., texts_for_labels[:50] for first 50)
                                    # and fontsize as needed.
                                    num_labels_to_show = min(len(texts_for_labels), 50) # Show up to 50 labels
                                    for i, txt_label in enumerate(texts_for_labels[:num_labels_to_show]):
                                        # Shorten the label if it's too long and add ellipsis
                                        display_label = (txt_label[:20] + '...') if len(txt_label) > 20 else txt_label
                                        ax.annotate(display_label, (coords_2d[i, 0], coords_2d[i, 1]), fontsize=6, alpha=0.75) # Smaller fontsize

                                    ax.set_title("2D PCA of RSS Title Embeddings (Chunks)")
                                    ax.set_xlabel("Principal Component 1")
                                    ax.set_ylabel("Principal Component 2")
                                    st.pyplot(fig) # Display the Matplotlib plot in Streamlit
                            with viz_col2:
                                if st.button("3D Scatter Plot", key="viz_3d_btn"):
                                    embeddings_to_visualize_3d = st.session_state.rss_embeddings_array
                                    texts_for_labels_3d = st.session_state.rss_embedded_texts

                                    # Reduce dimensionality to 3D using PCA
                                    pca_3d = PCA(n_components=3)
                                    coords_3d = pca_3d.fit_transform(embeddings_to_visualize_3d)

                                    # Create a Pandas DataFrame for Plotly Express
                                    df_3d_plot = pd.DataFrame(coords_3d, columns=["x", "y", "z"])
                                    df_3d_plot["label"] = [text[:30]+"..." for text in texts_for_labels_3d] # Shorten labels

                                    # Create an interactive 3D scatter plot using Plotly Express
                                    fig_3d = px.scatter_3d(
                                        df_3d_plot,
                                        x="x", y="y", z="z",
                                        text="label", # Text to display on hover or as labels
                                        height=700, # Height of the plot
                                        title="3D PCA of RSS Title Embeddings (Chunks)"
                                    )
                                    fig_3d.update_traces(marker=dict(size=3), textfont_size=8) # Adjust marker size and text
                                    st.plotly_chart(fig_3d, use_container_width=True) # Display Plotly chart

                        st.divider()
                        # --- Semantic Search Section ---
                        st.write("Semantic Search in Embedded Titles:")

                        # Use a form to allow both Enter and button click to trigger the search
                        with st.form(key="semantic_search_form"):
                            search_query = st.text_input("🔍 Enter search query for titles:", key="semantic_search_query")
                            search_submitted = st.form_submit_button("Search Titles")

                        if search_submitted:  # Triggered by either Enter or button click
                            if not search_query.strip():
                                st.warning("Please enter a search query.")
                            # Check if the index and metadata files exist
                            elif not os.path.exists("rss_titles.index") or not os.path.exists("rss_titles_meta.pkl"):
                                st.error("Index or metadata not found. Please ingest items first.")
                            else:
                                try:
                                    with st.spinner("Searching..."):
                                        # Embed the search query
                                        query_embedding = np.array(get_openai_embeddings([search_query])).astype("float32")
                                        # Load the FAISS index
                                        loaded_faiss_index = faiss.read_index("rss_titles.index")
                                        # Load the metadata (original items)
                                        with open("rss_titles_meta.pkl", "rb") as f_meta_load:
                                            loaded_metadata = pickle.load(f_meta_load)  # This is a list of original items
                                        # Load the mapping from chunks to original items
                                        chunk_to_original_map = st.session_state.get("rss_original_item_indices_for_chunks", [])
                                        embedded_chunks_texts = st.session_state.get("rss_embedded_texts", [])

                                        if not loaded_faiss_index.is_trained or loaded_faiss_index.ntotal == 0:
                                            st.warning("The index is empty or not trained. Please ingest items.")
                                        else:
                                            # Perform the search
                                            num_results_to_fetch = min(5, loaded_faiss_index.ntotal)  # Fetch top 5 or fewer
                                            distances, chunk_indices = loaded_faiss_index.search(query_embedding, num_results_to_fetch)

                                            if chunk_indices.size == 0 or len(chunk_indices[0]) == 0 or chunk_indices[0][0] == -1:
                                                st.info("No semantically similar items found for your query.")
                                            else:
                                                st.write("Search Results (most similar items based on title chunks):")
                                                displayed_original_indices = set()  # To avoid displaying the same original item multiple times

                                                for i, chunk_idx in enumerate(chunk_indices[0]):
                                                    if chunk_idx == -1:
                                                        continue  # Skip if FAISS indicates no neighbor

                                                    if chunk_idx < len(chunk_to_original_map):
                                                        original_item_idx = chunk_to_original_map[chunk_idx]

                                                        if original_item_idx not in displayed_original_indices and original_item_idx < len(loaded_metadata):
                                                            original_item = loaded_metadata[original_item_idx]

                                                            # Get the raw title and link
                                                            raw_title = original_item.get('Title', 'No Title Found')
                                                            link_to_display = original_item.get('Link', '#')

                                                       
                                                            # Safely get matched chunk text and escape it for display
                                                            raw_matched_chunk_text = embedded_chunks_texts[chunk_idx] if chunk_idx < len(embedded_chunks_texts) else ""
                                                            safe_display_matched_chunk = html.escape(raw_matched_chunk_text[:80]) if raw_matched_chunk_text else ""

                                                            # Step 1: Escape the raw title to neutralize any pre-existing HTML.
                                                            title_to_highlight_on = html.escape(raw_title)

                                                            highlighted_display_title = title_to_highlight_on
                                                            try:
                                                                # Step 2: Apply <mark> tags to the escaped title.
                                                                query_terms_for_highlight = [term for term in search_query.lower().split() if term and len(term) > 1]
                                                                for term_to_highlight in query_terms_for_highlight:
                                                                    # The pattern will search for the plain term within the escaped title.
                                                                    # m.group(0) will be the matched part of the (already escaped) title.
                                                                    pattern = re.compile(fr'(?<!\w){re.escape(term_to_highlight)}(?!\w)', re.IGNORECASE)
                                                                    highlighted_display_title = pattern.sub(
                                                                        lambda m: f"<mark>{html.escape(m.group(0))}</mark>", highlighted_display_title
                                                                    )
                                                            except Exception as e_highlight:
                                                                st.error(f"Highlighting error: {e_highlight}")
                                                                # Fallback to the escaped title without highlighting marks
                                                                highlighted_display_title = title_to_highlight_on

                                                            # Step 3: Construct the HTML output.
                                                            # `highlighted_display_title` now contains escaped original text + <mark> tags.
                                                            # `unsafe_allow_html=True` will render our <mark>s and the <a>, <br>, etc.
                                                            html_link_output = f"- <a href='{link_to_display}' target='_blank'>{highlighted_display_title}</a>"
                                                            if safe_display_matched_chunk:
                                                                html_link_output += f"<br><small><i>(Matched on chunk: \"{safe_display_matched_chunk}...\")</i></small>"

                                                            st.markdown(html_link_output, unsafe_allow_html=True)
                                                            displayed_original_indices.add(original_item_idx)
                                                        elif original_item_idx >= len(loaded_metadata):
                                                            st.warning(f"Metadata not found for original item index {original_item_idx} (from chunk index {chunk_idx}).")
                                                    else:
                                                        st.warning(f"Could not map chunk index {chunk_idx} to an original item.")
                                except Exception as e_search:
                                    st.error(f"An error occurred during semantic search: {e_search}")
                    else:
                        st.info("No items extracted from the feed to display or embed.")

                except KeyError as e_key:
                    st.error(f"Error accessing feed data (KeyError): {e_key}. The feed structure might be unexpected.")
                except Exception as e_display:
                    # Catch-all for other errors in this sub-tab
                    st.error(f"Error displaying or processing feed items: {e_display}")

            # --- Sub-Tab 3: Raw Data Display and Download ---
            with rss_subtab3:
                st.write("Raw JSON data (as text):")
                # Display the raw JSON data as a string in a text area
                raw_json_string = json.dumps(st.session_state.rss_json_data, indent=2) # Pretty print with indent
                st.text_area("JSON Output", raw_json_string, height=300)

                # Button to download the JSON data as a file
                st.download_button(
                    label="Download JSON Data",        # Button label
                    data=raw_json_string,              # Data to download
                    file_name="rss_feed_data.json",    # Default file name
                    mime="application/json",           # MIME type for JSON
                    key="rss_download_json_btn"
                )
        elif st.button("Clear Cached Feed Data", key="clear_feed_btn_col2", help="Click to remove the currently loaded feed data from this session."):
            if 'rss_json_data' in st.session_state: del st.session_state.rss_json_data
            if 'rss_embeddings_array' in st.session_state: del st.session_state.rss_embeddings_array
            if 'rss_embedded_texts' in st.session_state: del st.session_state.rss_embedded_texts
            if 'rss_original_item_indices_for_chunks' in st.session_state: del st.session_state.rss_original_item_indices_for_chunks
            # st.experimental_rerun() # Rerun the app to reflect the cleared state # OLD WAY
            st.rerun() # Rerun the app to reflect the cleared state # NEW WAY
        else:
            st.info("Fetch an RSS feed using the controls on the left to see its content here.")
# -----------------------------------------------------------------------------
with tab2: # Content for the second tab
    st.header("Cerebras LLM Completion") # Header for this tab

    # `st.form` creates a form that groups multiple input widgets.
    # When the submit button within the form is clicked, all widget values are sent together.
    # `clear_on_submit=False` means the input fields won't be cleared after submission.
    with st.form(key="cerebras_completion_form", clear_on_submit=False):
        st.write("Interact with Cerebras Large Language Models.")
        # Radio button to select the type of completion (Chat or Text)
        selected_completion_type = st.radio(
            "Select completion type:",
            options=["Chat Completion", "Text Completion"], # Available options
            horizontal=True # Display options side-by-side
        )

        # Text area for user to input their prompt
        user_prompt_input = st.text_area(
            "Enter your prompt:", # Label for the text area
            height=150,           # Height of the text area in pixels
            placeholder="e.g., Explain the theory of relativity in simple terms."
        )

        # Submit button for the form
        form_submitted = st.form_submit_button("Generate Completion")

        if form_submitted: 
            if user_prompt_input.strip(): 
                with st.spinner("Generating completion from Cerebras LLM..."):
                    try:
                        api_response = None
                        if selected_completion_type == "Chat Completion":
                            st.info("Requesting Chat Completion...")
                            api_response = get_chat_completion(user_prompt_input)
                        else: 
                            st.info("Requesting Text Completion...")
                            api_response = get_text_completion(user_prompt_input)

                        st.subheader("LLM Response:")
                        
                        # Debug information (can be removed in production)
                        with st.expander("Response Type Info"):
                            st.write(f"Response type: {type(api_response)}")
                        
                        # First try the attribute access method (recommended)
                        try:
                            if selected_completion_type == "Chat Completion":
                                # Try object-attribute access first (as per documentation)
                                response_text = api_response.choices[0].message.content
                                
                                # Display the response
                                st.markdown(response_text)
                                
                                # Optional: Display metadata
                                with st.expander("Response Metadata"):
                                    st.write(f"**Finish Reason:** {api_response.choices[0].finish_reason}")
                                    st.write(f"**Model:** {api_response.model}")
                                    if hasattr(api_response, 'usage'):
                                        st.write(f"**Usage:** {api_response.usage}")
                            
                            elif selected_completion_type == "Text Completion":
                                # For text completions - correct access based on API documentation
                                try:
                                    # The completion text should be in choices[0].text
                                    response_text = api_response.choices[0].text
                                    st.markdown(response_text)
                                    
                                    # Optional: Display metadata
                                    with st.expander("Response Metadata"):
                                        st.write(f"**Finish Reason:** {api_response.choices[0].finish_reason}")
                                        st.write(f"**Model:** {api_response.model}")
                                        if hasattr(api_response, 'usage'):
                                            st.write(f"**Usage:** {api_response.usage}")
                                except AttributeError:
                                    # If direct attribute access fails, try dictionary-based access
                                    if isinstance(api_response, dict) and "choices" in api_response and len(api_response["choices"]) > 0:
                                        if "text" in api_response["choices"][0]:
                                            response_text = api_response["choices"][0]["text"]
                                            st.markdown(response_text)
                                        else:
                                            st.error("Could not find 'text' in the completion response")
                                            st.json(api_response)
                                    else:
                                        st.error("Unexpected text completion response format")
                                        st.json(api_response)
                        
                        except (AttributeError, IndexError, TypeError) as attr_error:
                            st.warning(f"Could not access response with attribute notation: {attr_error}")
                            
                            # Fall back to dictionary-based access
                            response_text = None
                            
                            if isinstance(api_response, dict):
                                if selected_completion_type == "Chat Completion" and "choices" in api_response:
                                    choices = api_response["choices"]
                                    if choices and isinstance(choices, list) and len(choices) > 0:
                                        choice = choices[0]
                                        if isinstance(choice, dict) and "message" in choice:
                                            response_text = choice["message"].get("content", "")
                                        elif isinstance(choice, str) and "message=" in choice:
                                            import re
                                            content_match = re.search(r"content='([^']*)'", choice)
                                            if content_match:
                                                response_text = content_match.group(1)
                                elif selected_completion_type == "Text Completion" and "text" in api_response:
                                    response_text = api_response["text"]
                            
                            # Display the response if we found it
                            if response_text:
                                st.markdown(response_text)
                            else:
                                st.error("Could not extract response text using any known method")
                                st.json(api_response)

                    except Exception as e_cerebras:
                        st.error(f"An error occurred with the Cerebras API: {e_cerebras}")
            else:
                st.warning("Please enter a prompt before submitting.")
                
# -----------------------------------------------------------------------------
# TAB 3: Future Project (Placeholder)
# -----------------------------------------------------------------------------
with tab3: # Content for the third tab
    st.header("Future Project Area") # Header for this tab
    st.write("This space is reserved for your next awesome Streamlit project or feature!")
    st.info("Stay tuned for more updates...")

    # Initialize a flag in session state if it doesn't exist
    if 'tab3_balloons_shown_this_session' not in st.session_state:
        st.session_state.tab3_balloons_shown_this_session = False

    # Show balloons only if they haven't been shown yet in this session for this tab
    if not st.session_state.tab3_balloons_shown_this_session:
        st.balloons() # Fun Streamlit animation
        # Set the flag to True so balloons don't appear again in this session for this tab
        st.session_state.tab3_balloons_shown_this_session = True
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# FOOTER
# A common practice to add a small footer with credits or links.
# -----------------------------------------------------------------------------
st.divider() # Adds a horizontal line separator
# `st.caption` displays text in a smaller font, suitable for captions or footers.
# Using Markdown for the link.
st.caption("Created with Streamlit • View on [GitHub: nickpclarke/playground](https://github.com/nickpclarke/playground)")
# -----------------------------------------------------------------------------
