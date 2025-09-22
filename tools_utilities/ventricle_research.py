import os
import json
import requests
import xml.etree.ElementTree as ET

def get_api_key(service_name):
    """
    Retrieves the API key for a given service from the credentials file.
    """
    # Construct the path to the credentials file relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    credentials_path = os.path.join(script_dir, '../data/credentials/all_api_keys.json')

    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
            return credentials.get('services', {}).get(service_name, {}).get('api_key')
    except FileNotFoundError:
        print(f"Error: The credentials file was not found at {credentials_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: The credentials file at {credentials_path} is not valid JSON.")
        return None

def search_pubmed_central(query, api_key):
    """
    Searches PubMed Central for a given query and returns a list of article IDs.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query,
        "retmax": 10,  # Limiting to 10 results for this example
        "api_key": api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        root = ET.fromstring(response.content)
        id_list = [id_element.text for id_element in root.findall(".//Id")]
        return id_list
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return []

def fetch_pubmed_central_articles(id_list, api_key):
    """
    Fetches the full text of articles from PubMed Central given a list of IDs.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    articles = []
    for article_id in id_list:
        params = {
            "db": "pmc",
            "id": article_id,
            "retmode": "xml",
            "api_key": api_key
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            articles.append(response.text)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching article {article_id}: {e}")
    return articles

def save_articles(articles, directory="research_articles"):
    """
    Saves the fetched articles to a specified directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, article_content in enumerate(articles):
        file_path = os.path.join(directory, f"article_{i+1}.xml")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(article_content)
        print(f"Saved article to {file_path}")

if __name__ == "__main__":
    # Define the search query
    search_query = (
        '("ventricular system development" OR "ventriculogenesis") '
        'AND ("neural tube" OR "embryonic brain") '
        'AND ("cell migration" OR "apoptosis" OR "cell proliferation")'
    )

    # Get the PubMed Central API key
    pubmed_api_key = get_api_key("pubmed_central")

    if pubmed_api_key:
        # Search for articles
        article_ids = search_pubmed_central(search_query, pubmed_api_key)

        if article_ids:
            print(f"Found {len(article_ids)} articles.")
            # Fetch the full text of the articles
            # Note: Fetching full text might not be ideal for a quick overview.
            # parsing abstracts would be a better next step.
            # This is a placeholder for a more sophisticated pipeline.
            fetched_articles = fetch_pubmed_central_articles(article_ids, pubmed_api_key)
            save_articles(fetched_articles)

        else:
            print("No articles found for the given query.")
    else:
        print("Could not retrieve PubMed Central API key.")