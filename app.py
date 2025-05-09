import streamlit as st
import json
import requests
import xmltodict
import pandas as pd
from cerebras_client import get_chat_completion, get_text_completion

st.set_page_config(
    page_title="Multi-Tool App",
    page_icon="🛠️",
    layout="wide"
)

# --- Main app header ---
st.title("🛠️ Multi-Tool Dashboard")

# --- Create tabs for different applications ---
tab1, tab2, tab3 = st.tabs(["RSS to JSON", "Cerebras Completion", "Future Project"])

# --- TAB 1: RSS to JSON Converter ---
with tab1:
    st.header("RSS to JSON Converter")
    
    # RSS to JSON functionality
    def rss_to_json(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            xml_content = response.content
            json_data = xmltodict.parse(xml_content)
            return json_data, None
        except requests.RequestException as e:
            return None, f"Error fetching RSS feed: {e}"
        except xmltodict.ParsingError as e:
            return None, f"Error parsing XML: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"
    
    # Default RSS feeds
    default_feeds = {
        "Google News": "https://news.google.com/rss",
        "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
        "CNN News": "http://rss.cnn.com/rss/edition.rss",
        "NASA Breaking News": "https://www.nasa.gov/rss/dyn/breaking_news.rss"
    }
    
    # Two columns layout inside the tab
    col1, col2 = st.columns([1, 2])
    
    with col1:
        feed_option = st.radio(
            "Choose an RSS feed source:",
            ["Select from defaults", "Enter custom URL"],
            key="rss_feed_option"
        )
        
        if feed_option == "Select from defaults":
            selected_feed = st.selectbox("Select a feed:", list(default_feeds.keys()))
            rss_url = default_feeds[selected_feed]
        else:
            rss_url = st.text_input("Enter RSS feed URL:", "https://news.google.com/rss")
        
        if st.button("Convert to JSON", key="rss_convert_btn"):
            if rss_url:
                with st.spinner("Fetching and converting feed..."):
                    json_data, error = rss_to_json(rss_url)
                    
                    if error:
                        st.error(error)
                    else:
                        st.session_state.rss_json_data = json_data
                        st.success("RSS feed converted successfully!")
            else:
                st.error("Please enter a valid URL")
    
    with col2:
        if 'rss_json_data' in st.session_state:
            rss_subtab1, rss_subtab2, rss_subtab3 = st.tabs(["JSON View", "Feed Items", "Raw Data"])
            
            with rss_subtab1:
                st.json(st.session_state.rss_json_data)
            
            with rss_subtab2:
                try:
                    if 'rss' in st.session_state.rss_json_data:
                        items = st.session_state.rss_json_data['rss']['channel']['item']
                    elif 'feed' in st.session_state.rss_json_data:
                        items = st.session_state.rss_json_data['feed']['entry']
                    else:
                        items = []
                        st.warning("Could not identify feed item structure")
                    
                    if items:
                        if not isinstance(items, list):
                            items = [items]
                        
                        df = pd.DataFrame([{
                            'Title': item.get('title', 'No Title'),
                            'Link': item.get('link', item.get('guid', 'No Link'))
                        } for item in items])
                        
                        st.dataframe(df)
                except Exception as e:
                    st.error(f"Error displaying feed items: {e}")
            
            with rss_subtab3:
                st.text(json.dumps(st.session_state.rss_json_data, indent=2))
                
                st.download_button(
                    "Download JSON",
                    json.dumps(st.session_state.rss_json_data, indent=2),
                    "rss_feed.json",
                    "application/json",
                    key="rss_download_btn"
                )

# --- TAB 2: Cerebras Completion ---
with tab2:
    st.header("Cerebras Completion")
    
    # Form for Cerebras completion
    with st.form(key="completion_form", clear_on_submit=False):
        # Add radio button for selecting completion type
        completion_type = st.radio(
            "Select completion type:",
            options=["Chat Completion", "Text Completion"],
            horizontal=True
        )
        
        user_input = st.text_input("Enter your prompt:")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            if user_input:
                with st.spinner("Generating completion..."):
                    try:
                        # Call the appropriate completion function based on user selection
                        if completion_type == "Chat Completion":
                            response = get_chat_completion(user_input)
                            st.info("Chat Completion:")
                            st.write(response)
                        else:  # Text Completion
                            response = get_text_completion(user_input)
                            st.info("Text Completion:")
                            st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a prompt.")

# --- TAB 3: Future Project (Placeholder) ---
with tab3:
    st.header("Future Project")
    st.write("This space is reserved for your next awesome project!")
    st.info("Coming soon...")

# --- Footer ---
st.divider()
st.caption("Created with Streamlit • GitHub: [your-username/your-repo](https://github.com/your-username/your-repo)")