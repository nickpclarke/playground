import streamlit as st
import json
import requests
import xmltodict
import pandas as pd

def rss_to_json(url):
    """
    Fetch an RSS feed from a URL and convert it to a JSON object
    """
    try:
        # Fetch the RSS feed
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse XML to Python dict
        xml_content = response.content
        json_data = xmltodict.parse(xml_content)
        
        return json_data, None
    except requests.RequestException as e:
        error_msg = f"Error fetching RSS feed: {e}"
        return None, error_msg
    except xmltodict.ParsingError as e:
        error_msg = f"Error parsing XML: {e}"
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        return None, error_msg

st.title("RSS to JSON Converter")

# Default RSS feeds in a dropdown
default_feeds = {
    "Google News": "https://news.google.com/rss",
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "CNN News": "http://rss.cnn.com/rss/edition.rss",
    "NASA Breaking News": "https://www.nasa.gov/rss/dyn/breaking_news.rss"
}

# Two columns layout
col1, col2 = st.columns([1, 2])

with col1:
    # Option to select from default feeds or enter custom URL
    feed_option = st.radio(
        "Choose an RSS feed source:",
        ["Select from defaults", "Enter custom URL"]
    )
    
    if feed_option == "Select from defaults":
        selected_feed = st.selectbox("Select a feed:", list(default_feeds.keys()))
        rss_url = default_feeds[selected_feed]
    else:
        rss_url = st.text_input("Enter RSS feed URL:", "https://news.google.com/rss")
    
    # Submit button
    if st.button("Convert to JSON"):
        if rss_url:
            with st.spinner("Fetching and converting feed..."):
                json_data, error = rss_to_json(rss_url)
                
                if error:
                    st.error(error)
                else:
                    # Store in session state so we can access it in the other column
                    st.session_state.json_data = json_data
                    st.success("RSS feed converted successfully!")
        else:
            st.error("Please enter a valid URL")

# Display results in the second column
with col2:
    if 'json_data' in st.session_state:
        # Add tabs for different views
        tab1, tab2, tab3 = st.tabs(["JSON View", "Feed Items", "Raw Data"])
        
        with tab1:
            st.json(st.session_state.json_data)
        
        with tab2:
            try:
                # Try to extract feed items - this depends on RSS structure
                if 'rss' in st.session_state.json_data:
                    items = st.session_state.json_data['rss']['channel']['item']
                elif 'feed' in st.session_state.json_data:
                    items = st.session_state.json_data['feed']['entry']
                else:
                    items = []
                    st.warning("Could not identify feed item structure")
                
                if items:
                    if not isinstance(items, list):
                        items = [items]  # Handle single item case
                    
                    # Create a simple table of titles and links
                    df = pd.DataFrame([{
                        'Title': item.get('title', 'No Title'),
                        'Link': item.get('link', item.get('guid', 'No Link'))
                    } for item in items])
                    
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Error displaying feed items: {e}")
        
        with tab3:
            st.text(json.dumps(st.session_state.json_data, indent=2))
            
            # Download button for JSON
            st.download_button(
                "Download JSON",
                json.dumps(st.session_state.json_data, indent=2),
                "rss_feed.json",
                "application/json"
            )