import numpy as np
import time
import streamlit as st
from openai_client import get_openai_embeddings
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import faiss
import re

def generate_embeddings(texts_to_embed, show_progress=False):
    """
    Generate embeddings for the text chunks.
    
    Args:
        texts_to_embed (list): List of text chunks to embed
        show_progress (bool): Whether to display progress in Streamlit
        
    Returns:
        list: List of embedding vectors
    """
    embeddings = []
    
    # Show progress if requested
    embedding_status = None
    embedding_progress = None
    if show_progress:
        embedding_status = st.empty()
        embedding_progress = st.progress(0)
    
    for i, text in enumerate(texts_to_embed):
        if show_progress:
            progress = (i + 1) / len(texts_to_embed)
            embedding_progress.progress(progress)
            embedding_status.text(f"Processing embedding {i+1}/{len(texts_to_embed)}")
        
        # Add a small delay to prevent rate limiting
        if i > 0 and i % 10 == 0:
            time.sleep(1)  # Short delay every 10 requests
        
        # Get embedding with better error handling
        try:
            embedding = get_openai_embeddings(text)
            if embedding is not None:
                embeddings.append(embedding)
            elif show_progress:
                st.warning(f"Could not generate embedding for chunk {i+1}")
                # Try again with a simplified version of the text
                simplified_text = text[:1000]  # Limit to 1000 chars
                if embedding_status:
                    embedding_status.text(f"Retrying with simplified text for chunk {i+1}")
                embedding = get_openai_embeddings(simplified_text)
                if embedding is not None:
                    embeddings.append(embedding)
                    if show_progress:
                        st.info(f"Successfully generated embedding for simplified chunk {i+1}")
                elif show_progress:
                    st.error(f"Failed to generate embedding for chunk {i+1} even after simplification")
        except Exception as chunk_error:
            if show_progress:
                st.error(f"Error on chunk {i+1}: {str(chunk_error)}")
    
    return embeddings

def visualize_embeddings_2d(embeddings_array, texts, title_suffix=""):
    """
    Create a 2D visualization of embeddings with text labels.
    
    Args:
        embeddings_array (np.ndarray): The embeddings to visualize
        texts (list): The texts corresponding to the embeddings
        title_suffix (str): Optional suffix for the visualization title
    """
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)
    
    # Create a DataFrame for plotting
    df_viz = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    
    # Add a column with shortened text for tooltips and labels
    df_viz['text'] = [text[:50] + "..." if len(text) > 50 else text for text in texts]
    
    # Add a column with very short text for labels (first 20 chars)
    df_viz['short_text'] = [text[:20] + "..." if len(text) > 20 else text for text in texts]
    
    # Create an interactive scatter plot with Plotly that includes text labels
    fig = px.scatter(
        df_viz, x='x', y='y', 
        hover_data=['text'],  # Full tooltip text (50 chars)
        text='short_text',    # Text to display on points (20 chars)
        labels={'x': 'Component 1', 'y': 'Component 2'},
        title=f"2D PCA Projection of Embeddings {title_suffix}"
    )
    
    # Configure text display settings
    fig.update_traces(
        textposition='top center',  # Position text above points
        textfont=dict(size=10),     # Smaller font size
        marker=dict(size=8)         # Slightly larger markers
    )
    
    # Improve layout to handle text better
    fig.update_layout(
        height=600,                 # Taller plot to accommodate labels
        margin=dict(l=20, r=20, t=50, b=20),  # Adjust margins
        showlegend=False            # Hide legend for cleaner view
    )
    
    st.write(f"### 2D Projection of Embeddings {title_suffix}")
    st.plotly_chart(fig, use_container_width=True)

def visualize_embeddings_3d(embeddings_array, texts, title_suffix=""):
    """
    Create a 3D visualization of embeddings with text labels.
    
    Args:
        embeddings_array (np.ndarray): The embeddings to visualize
        texts (list): The texts corresponding to the embeddings
        title_suffix (str): Optional suffix for the visualization title
    """
    # Reduce dimensionality for visualization
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings_array)
    
    # Create a DataFrame for plotting
    df_viz = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    
    # Add a column with shortened text for tooltips
    df_viz['text'] = [text[:50] + "..." if len(text) > 50 else text for text in texts]
    
    # Add a column with very short text for labels
    df_viz['short_text'] = [text[:15] + "..." if len(text) > 15 else text for text in texts]
    
    # Create an interactive 3D scatter plot with Plotly that includes text labels
    fig = px.scatter_3d(
        df_viz, x='x', y='y', z='z', 
        hover_data=['text'],  # Full tooltip text (50 chars)
        text='short_text',    # Text to display on points (15 chars - shorter for 3D)
        labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
        title=f"3D PCA Projection of Embeddings {title_suffix}"
    )
    
    # Configure text display settings in 3D
    fig.update_traces(
        textposition='top center',  # Position text above points
        textfont=dict(size=8),      # Smaller font size for 3D
        marker=dict(size=5)         # Smaller markers for 3D
    )
    
    # Improve 3D layout
    fig.update_layout(
        height=700,                 # Taller plot for 3D
        scene=dict(                 # Adjust 3D scene properties
            aspectmode='cube',      # Equal aspect ratio
            annotations=[]          # Initialize empty annotations list
        ),
        margin=dict(l=0, r=0, t=50, b=0)  # Tight margins for more space
    )
    
    st.write(f"### 3D Projection of Embeddings {title_suffix}")
    st.plotly_chart(fig, use_container_width=True)

def perform_semantic_search(query, embeddings_array, texts_to_embed, original_item_indices, simplified_items):
    """
    Perform semantic search on the embeddings.
    
    Args:
        query (str): The search query
        embeddings_array (np.ndarray): The embeddings to search through
        texts_to_embed (list): The texts corresponding to the embeddings
        original_item_indices (list): Mapping from chunk index to item index
        simplified_items (list): List of dictionaries containing feed items
    """
    query_embedding = get_openai_embeddings(query)
    
    if query_embedding is not None:
        # Create a FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Search for similar embeddings
        k = min(5, len(embeddings_array))  # Number of results to return
        query_embedding_array = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_embedding_array, k)
        
        # Prepare keywords for highlighting
        keywords = [word.lower() for word in query.split() if len(word) > 2]
        
        # Display search results
        st.write("### Search Results:")
        for i, idx in enumerate(indices[0]):
            if idx < len(texts_to_embed):
                original_item_idx = original_item_indices[idx]
                if original_item_idx < len(simplified_items):
                    item = simplified_items[original_item_idx]
                    
                    # Calculate relevance score
                    relevance_score = 1/(1+distances[0][i])
                    
                    # Display item title with highlighting
                    title = item['title']
                    title_highlighted = title
                    for keyword in keywords:
                        if keyword in title.lower():
                            # Use CSS to highlight matched keywords in the title
                            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                            title_highlighted = pattern.sub(
                                lambda m: f"<span style='background-color: yellow; color: black;'>{m.group(0)}</span>", 
                                title_highlighted
                            )
                    
                    st.markdown(f"**{i+1}. {title_highlighted}**", unsafe_allow_html=True)
                    st.markdown(f"*Relevance score: {relevance_score:.2f}*")
                    
                    # Display description with highlighting
                    description = item['description']
                    description_excerpt = description[:200] + "..." if len(description) > 200 else description
                    
                    # Highlight keywords in description
                    for keyword in keywords:
                        if keyword in description_excerpt.lower():
                            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                            description_excerpt = pattern.sub(
                                lambda m: f"<span style='background-color: yellow; color: black;'>{m.group(0)}</span>", 
                                description_excerpt
                            )
                    
                    st.markdown(description_excerpt, unsafe_allow_html=True)
                    st.write(f"[Read more]({item['link']})")
                    st.divider()
        
        return True
    else:
        st.error("Failed to generate embedding for search query.")
        return False