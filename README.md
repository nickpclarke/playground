# Multi-Tool Streamlit Application

This project is a versatile web application built using Streamlit. It includes tools for RSS feed parsing, semantic search, data visualization, and interaction with Cerebras Large Language Models (LLMs). The app is organized into three main tabs for different functionalities.

## Project Structure

```
streamlit-chat-app
├── app.py                # Main entry point for the Streamlit web application
├── cerebras_client.py    # Contains the Cerebras client initialization and chat functions
├── openai_client.py      # Contains functions for generating OpenAI embeddings
├── requirements.txt      # Lists the project dependencies
├── README.md             # Documentation for the project
└── .streamlit
    └── config.toml       # Configuration settings for the Streamlit application
```

## Features

### Tab 1: RSS to JSON Converter & Semantic Search
- **RSS Feed Parsing**:
  - Fetch RSS feeds from predefined sources (e.g., Google News, BBC News) or custom URLs.
  - Convert RSS feed data into JSON format for further processing.
- **Feed Item Embedding**:
  - Extract titles from RSS feed items and split them into chunks for embedding.
  - Use OpenAI embeddings to generate vector representations of the titles.
  - Store embeddings in a FAISS index for efficient similarity search.
- **Semantic Search**:
  - Perform semantic search on embedded titles using a query.
  - Highlight matching terms in the search results.
  - Display links to the original articles and matched chunks.
- **Data Visualization**:
  - Visualize embedded titles using 2D and 3D scatter plots (PCA projection).
  - Interactive 3D plots created with Plotly.

### Tab 2: Cerebras LLM Completion
- **Text and Chat Completions**:
  - Interact with Cerebras Large Language Models (LLMs) to generate text or chat completions.
  - Input prompts and receive responses from the model.
  - Display responses in a user-friendly format.

### Tab 3: Future Project Area
- Placeholder for future features or projects.
- Includes a fun balloon animation that appears the first time the tab is accessed in a session.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd streamlit-chat-app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables**:
   - Set the `CEREBRAS_API_KEY` environment variable with your Cerebras API key.
   - If using OpenAI embeddings, set the `OPENAI_API_KEY` environment variable.

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Tab 1: RSS to JSON & Semantic Search
1. Select a predefined RSS feed or enter a custom URL.
2. Click "Convert to JSON" to fetch and parse the feed.
3. View the feed data in JSON format, as a table, or as raw text.
4. Embed feed item titles for semantic search and visualization.
5. Perform semantic search by entering a query and view highlighted results.
6. Visualize embeddings in 2D or 3D scatter plots.

### Tab 2: Cerebras LLM Completion
1. Select the type of completion (Chat or Text).
2. Enter a prompt in the text area.
3. Click "Generate Completion" to receive a response from the Cerebras LLM.

### Tab 3: Future Project Area
- Placeholder for upcoming features or projects.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.