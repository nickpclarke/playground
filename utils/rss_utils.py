import requests
import xmltodict
import re
import pandas as pd

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

def extract_feed_items(json_data):
    """
    Extract items from an RSS feed JSON structure, handling various formats.
    
    Args:
        json_data (dict): The parsed RSS JSON data
        
    Returns:
        list: List of feed items or empty list if none found
    """
    items = []
    try:
        # Handle RSS format
        if 'rss' in json_data and 'channel' in json_data['rss'] and 'item' in json_data['rss']['channel']:
            items = json_data['rss']['channel']['item']
            if not isinstance(items, list):  # Handle case where there's only one item
                items = [items]
        # Handle Atom format
        elif 'feed' in json_data and 'entry' in json_data['feed']:
            items = json_data['feed']['entry']
            if not isinstance(items, list):  # Handle case where there's only one entry
                items = [items]
    except Exception:
        pass
    return items

def create_items_dataframe(items):
    """
    Create a pandas DataFrame from RSS feed items.
    
    Args:
        items (list): List of feed item dictionaries
        
    Returns:
        pandas.DataFrame: DataFrame of simplified feed items
    """
    simplified_items = []
    for i, item in enumerate(items):
        simplified_item = {}
        
        # Extract title
        if 'title' in item:
            if isinstance(item['title'], dict) and '#text' in item['title']:
                simplified_item['title'] = item['title']['#text']
            else:
                simplified_item['title'] = str(item['title'])
        else:
            simplified_item['title'] = f"Item {i+1}"
            
        # Extract description or content
        if 'description' in item:
            description = item['description']
            if isinstance(description, dict) and '#text' in description:
                description = description['#text']
            simplified_item['description'] = re.sub(r'<[^>]+>', '', str(description))
        elif 'content' in item:
            content = item['content']
            if isinstance(content, dict) and '#text' in content:
                content = content['#text']
            simplified_item['description'] = re.sub(r'<[^>]+>', '', str(content))
        else:
            simplified_item['description'] = "No description available"
            
        # Extract link
        if 'link' in item:
            if isinstance(item['link'], dict) and '@href' in item['link']:
                simplified_item['link'] = item['link']['@href']
            elif isinstance(item['link'], dict) and '#text' in item['link']:
                simplified_item['link'] = item['link']['#text']
            elif isinstance(item['link'], str):
                simplified_item['link'] = item['link']
            else:
                simplified_item['link'] = "No link available"
        else:
            simplified_item['link'] = "No link available"
            
        # Extract publication date
        if 'pubDate' in item:
            simplified_item['pubDate'] = item['pubDate']
        elif 'published' in item:
            simplified_item['pubDate'] = item['published']
        else:
            simplified_item['pubDate'] = "Unknown date"
            
        simplified_items.append(simplified_item)
    
    return pd.DataFrame(simplified_items)