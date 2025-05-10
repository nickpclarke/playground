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
    # Add debugging
    print(f"Chunking text with max_tokens={max_tokens}, text length: {len(text.split())} words")
    
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

    # Add debugging
    print(f"Created {len(chunks)} chunks")
    return chunks

def create_text_chunks(simplified_items, chunk_size, show_progress=False):
    """
    Create text chunks from the feed items.
    
    Args:
        simplified_items (list): List of dictionaries containing feed items
        chunk_size (int): Maximum number of words per chunk
        show_progress (bool): Whether to display progress in Streamlit
        
    Returns:
        tuple: (texts_to_embed, original_item_indices)
    """
    import streamlit as st
    
    texts_to_embed = []
    original_item_indices = []
    
    # Show progress if requested (for Streamlit UI)
    progress_bar = None
    chunk_status = None
    if show_progress:
        progress_bar = st.progress(0)
        chunk_status = st.empty()
    
    for idx, item in enumerate(simplified_items):
        # Update progress if showing
        if show_progress:
            progress = (idx + 1) / len(simplified_items)
            progress_bar.progress(progress)
            chunk_status.text(f"Processing item {idx+1}/{len(simplified_items)}")
        
        # Combine title and description for richer context
        text_to_chunk = f"{item['title']} {item['description']}"
        chunks = chunk_text(text_to_chunk, max_tokens=chunk_size)
        
        for chunk in chunks:
            texts_to_embed.append(chunk)
            original_item_indices.append(idx)
    
    if show_progress and chunk_status:
        chunk_status.text(f"Created {len(texts_to_embed)} chunks from {len(simplified_items)} items")
    
    return texts_to_embed, original_item_indices