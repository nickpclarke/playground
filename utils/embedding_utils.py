import numpy as np
import time
import streamlit as st
from openai_client import get_openai_embeddings
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import faiss
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 1
CACHE_TTL = 3600  # 1 hour

@st.cache_data(ttl=CACHE_TTL)
def generate_embeddings_batch(texts: List[str], batch_size: int = BATCH_SIZE, show_progress: bool = False) -> List[Optional[List[float]]]:
    """
    Generate embeddings for a batch of texts with caching and retry logic.
    
    Args:
        texts (List[str]): List of texts to embed
        batch_size (int): Number of texts to process in each batch
        show_progress (bool): Whether to show progress in Streamlit
        
    Returns:
        List[Optional[List[float]]]: List of embedding vectors
    """
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Initialize progress tracking
    progress_bar = None
    status_text = None
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        # Update progress
        if show_progress:
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}")
        
        # Process batch with retries
        for retry in range(MAX_RETRIES):
            try:
                batch_embeddings = []
                for i, text in enumerate(batch_texts):
                    # Update item progress within batch
                    if show_progress:
                        item_progress = (i + 1) / len(batch_texts)
                        status_text.text(f"Processing batch {batch_idx + 1}/{total_batches} - Item {i + 1}/{len(batch_texts)}")
                    
                    # Truncate text if too long
                    if len(text) > 8000:  # OpenAI's limit
                        text = text[:8000]
                    
                    embedding = get_openai_embeddings(text)
                    if embedding is not None:
                        batch_embeddings.append(embedding)
                    else:
                        batch_embeddings.append(None)
                
                embeddings.extend(batch_embeddings)
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed (attempt {retry + 1}/{MAX_RETRIES}): {str(e)}")
                if retry < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (retry + 1))  # Exponential backoff
                else:
                    # On final retry, add None for failed embeddings
                    embeddings.extend([None] * len(batch_texts))
    
    # Clear progress indicators
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    return embeddings

def visualize_embeddings_2d(embeddings_array: np.ndarray, texts: List[str], title_suffix: str = "") -> None:
    """
    Create a 2D visualization of embeddings with improved text handling.
    
    Args:
        embeddings_array (np.ndarray): The embeddings to visualize
        texts (List[str]): The texts corresponding to the embeddings
        title_suffix (str): Optional suffix for the visualization title
    """
    if len(embeddings_array) == 0:
        st.warning("No embeddings to visualize")
        return
        
    try:
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        # Create a DataFrame for plotting
        df_viz = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        df_viz['text'] = [text[:50] + "..." if len(text) > 50 else text for text in texts]
        df_viz['short_text'] = [text[:20] + "..." if len(text) > 20 else text for text in texts]
        
        # Create interactive scatter plot
        fig = px.scatter(
            df_viz, x='x', y='y',
            hover_data=['text'],
            text='short_text',
            labels={'x': 'Component 1', 'y': 'Component 2'},
            title=f"2D PCA Projection of Embeddings {title_suffix}"
        )
        
        # Configure text display
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(size=8)
        )
        
        # Improve layout
        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in 2D visualization: {str(e)}")
        st.error("Failed to create 2D visualization")

def visualize_embeddings_3d(embeddings_array: np.ndarray, texts: List[str], title_suffix: str = "") -> None:
    """
    Create a 3D visualization of embeddings with improved text handling.
    
    Args:
        embeddings_array (np.ndarray): The embeddings to visualize
        texts (List[str]): The texts corresponding to the embeddings
        title_suffix (str): Optional suffix for the visualization title
    """
    if len(embeddings_array) == 0:
        st.warning("No embeddings to visualize")
        return
        
    try:
        # Reduce dimensionality for visualization
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings_array)
        
        # Create a DataFrame for plotting
        df_viz = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
        df_viz['text'] = [text[:50] + "..." if len(text) > 50 else text for text in texts]
        df_viz['short_text'] = [text[:15] + "..." if len(text) > 15 else text for text in texts]
        
        # Create interactive 3D scatter plot
        fig = px.scatter_3d(
            df_viz, x='x', y='y', z='z',
            hover_data=['text'],
            text='short_text',
            labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
            title=f"3D PCA Projection of Embeddings {title_suffix}"
        )
        
        # Configure text display
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=8),
            marker=dict(size=5)
        )
        
        # Improve 3D layout
        fig.update_layout(
            height=700,
            scene=dict(
                aspectmode='cube',
                annotations=[]
            ),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in 3D visualization: {str(e)}")
        st.error("Failed to create 3D visualization")

@st.cache_resource
def create_faiss_index(embeddings_array: np.ndarray) -> faiss.Index:
    """
    Create and cache a FAISS index for the embeddings.
    
    Args:
        embeddings_array (np.ndarray): The embeddings to index
        
    Returns:
        faiss.Index: The FAISS index
    """
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array.astype('float32'))
    return index

def perform_semantic_search(
    query: str,
    embeddings_array: np.ndarray,
    texts: List[str],
    original_item_indices: List[int],
    simplified_items: List[Dict[str, Any]],
    top_k: int = 5
) -> bool:
    """
    Perform semantic search with improved relevance scoring and highlighting.
    
    Args:
        query (str): The search query
        embeddings_array (np.ndarray): The embeddings to search through
        texts (List[str]): The texts corresponding to the embeddings
        original_item_indices (List[int]): Mapping from chunk index to item index
        simplified_items (List[Dict[str, Any]]): List of dictionaries containing feed items
        top_k (int): Number of results to return
        
    Returns:
        bool: True if search was successful, False otherwise
    """
    try:
        # Get query embedding
        query_embedding = get_openai_embeddings(query)
        if query_embedding is None:
            st.error("Failed to generate embedding for search query")
            return False
            
        # Create or get cached FAISS index
        index = create_faiss_index(embeddings_array)
        
        # Search for similar embeddings
        k = min(top_k, len(embeddings_array))
        query_embedding_array = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_embedding_array, k)
        
        # Prepare keywords for highlighting
        keywords = [word.lower() for word in query.split() if len(word) > 2]
        
        # Display search results
        st.write("### Search Results")
        for i, idx in enumerate(indices[0]):
            if idx >= len(texts):
                continue
                
            original_item_idx = original_item_indices[idx]
            if original_item_idx >= len(simplified_items):
                continue
                
            item = simplified_items[original_item_idx]
            
            # Calculate relevance score (normalized distance)
            max_distance = np.max(distances[0])
            relevance_score = 1 - (distances[0][i] / max_distance if max_distance > 0 else 0)
            
            # Highlight text
            title = highlight_text(item['title'], keywords)
            description = highlight_text(
                item['description'][:200] + "..." if len(item['description']) > 200 else item['description'],
                keywords
            )
            
            # Display result
            st.markdown(f"**{i+1}. {title}**", unsafe_allow_html=True)
            st.markdown(f"*Relevance score: {relevance_score:.2f}*")
            st.markdown(description, unsafe_allow_html=True)
            st.write(f"[Read more]({item['link']})")
            st.divider()
            
        return True
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        st.error("An error occurred during search")
        return False

def highlight_text(text: str, keywords: List[str]) -> str:
    """
    Highlight keywords in text with HTML formatting.
    
    Args:
        text (str): The text to highlight
        keywords (List[str]): List of keywords to highlight
        
    Returns:
        str: Text with highlighted keywords
    """
    highlighted_text = text
    for keyword in keywords:
        if keyword in highlighted_text.lower():
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(
                lambda m: f"<span style='background-color: yellow; color: black;'>{m.group(0)}</span>",
                highlighted_text
            )
    return highlighted_text