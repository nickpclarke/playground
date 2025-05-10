import requests
import xmltodict
import re
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache
from datetime import datetime, timedelta
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 5
REQUEST_TIMEOUT = 10
CACHE_TTL = 3600  # 1 hour

@st.cache_data(ttl=CACHE_TTL)
def fetch_rss_feed(url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches an RSS feed with caching and improved error handling.
    
    Args:
        url (str): The URL of the RSS feed.
        
    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[str]]: A tuple containing the parsed data
        and error message if any.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'xml' not in content_type and 'rss' not in content_type and 'atom' not in content_type:
            logger.warning(f"Unexpected content type for {url}: {content_type}")
        
        return xmltodict.parse(response.content, dict_constructor=dict), None
        
    except requests.exceptions.Timeout:
        return None, f"Error: Request to {url} timed out after {REQUEST_TIMEOUT} seconds."
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching RSS feed: {str(e)}"
    except xmltodict.ParsingError as e:
        return None, f"Error parsing XML: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {str(e)}")
        return None, f"An unexpected error occurred: {str(e)}"

def batch_fetch_rss_feeds(urls: List[str]) -> List[Tuple[Dict[str, Any], Optional[str]]]:
    """
    Fetches multiple RSS feeds concurrently.
    
    Args:
        urls (List[str]): List of RSS feed URLs.
        
    Returns:
        List[Tuple[Dict[str, Any], Optional[str]]]: List of tuples containing parsed data and errors.
    """
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(fetch_rss_feed, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data, error = future.result()
                results.append((data, error))
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                results.append((None, str(e)))
    return results

def extract_feed_items(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract items from an RSS feed JSON structure with improved format handling.
    
    Args:
        json_data (Dict[str, Any]): The parsed RSS JSON data
        
    Returns:
        List[Dict[str, Any]]: List of feed items
    """
    if not json_data:
        return []
        
    items = []
    try:
        # Handle RSS format
        if 'rss' in json_data and 'channel' in json_data['rss']:
            channel = json_data['rss']['channel']
            if 'item' in channel:
                items = channel['item']
                if not isinstance(items, list):
                    items = [items]
                    
        # Handle Atom format
        elif 'feed' in json_data and 'entry' in json_data['feed']:
            items = json_data['feed']['entry']
            if not isinstance(items, list):
                items = [items]
                
        # Handle JSON Feed format
        elif 'items' in json_data:
            items = json_data['items']
            if not isinstance(items, list):
                items = [items]
                
    except Exception as e:
        logger.error(f"Error extracting feed items: {str(e)}")
        
    return items

def create_items_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from RSS feed items with improved data cleaning.
    
    Args:
        items (List[Dict[str, Any]]): List of feed item dictionaries
        
    Returns:
        pd.DataFrame: DataFrame of cleaned feed items
    """
    if not items:
        return pd.DataFrame()
        
    simplified_items = []
    for i, item in enumerate(items):
        try:
            simplified_item = {
                'title': extract_text(item.get('title', f"Item {i+1}")),
                'description': clean_html(extract_text(item.get('description') or item.get('content', "No description available"))),
                'link': extract_link(item.get('link', "No link available")),
                'pubDate': parse_date(item.get('pubDate') or item.get('published') or "Unknown date"),
                'guid': item.get('guid', {}).get('#text', str(i)) if isinstance(item.get('guid'), dict) else item.get('guid', str(i))
            }
            simplified_items.append(simplified_item)
        except Exception as e:
            logger.error(f"Error processing item {i}: {str(e)}")
            continue
    
    df = pd.DataFrame(simplified_items)
    if not df.empty:
        df['pubDate'] = pd.to_datetime(df['pubDate'], errors='coerce')
        df = df.sort_values('pubDate', ascending=False)
    
    return df

def extract_text(value: Any) -> str:
    """Extract text from various input formats."""
    if isinstance(value, dict):
        return value.get('#text', '')
    return str(value)

def extract_link(link: Any) -> str:
    """Extract link from various input formats."""
    if isinstance(link, dict):
        return link.get('@href') or link.get('#text', '')
    return str(link)

def clean_html(text: str) -> str:
    """Clean HTML tags and normalize whitespace."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_date(date_str: str) -> datetime:
    """Parse various date formats."""
    try:
        return pd.to_datetime(date_str)
    except:
        return datetime.now()