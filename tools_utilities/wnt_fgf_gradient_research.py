import os
import requests
import xml.etree.ElementTree as ET
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
MAX_RESULTS = 10

# Search terms
SEARCH_TERMS = {
    "WNT": '"wnt" AND "neural tube" AND ("gradient" OR "patterning")',
    "FGF": '"fibroblast growth factor" AND "neural tube" AND ("gradient" OR "patterning")'
}

def search_pubmed(query: str, max_results: int) -> list:
    """Search PubMed for a given query and return a list of article IDs."""
    search_url = f"{BASE_URL}esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "usehistory": "y"
    }
    logging.info(f"Searching PubMed with query: {query}")
    response = requests.get(search_url, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    id_list = [id_elem.text for id_elem in root.findall(".//Id")]
    logging.info(f"Found {len(id_list)} articles.")
    return id_list

def fetch_articles(id_list: list) -> str:
    """Fetch article details from PubMed for a list of IDs."""
    if not id_list:
        return ""
    
    fetch_url = f"{BASE_URL}efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    logging.info(f"Fetching details for {len(id_list)} articles.")
    response = requests.get(fetch_url, params=params)
    response.raise_for_status()
    return response.text

def save_articles(article_data: str, output_dir: str):
    """Save fetched articles to individual XML files."""
    if not article_data:
        logging.warning("No article data to save.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = ET.fromstring(article_data)
    articles = root.findall(".//PubmedArticle")
    for i, article in enumerate(articles):
        pmid_element = article.find(".//PMID")
        if pmid_element is not None and pmid_element.text:
            pmid = pmid_element.text
            filename = os.path.join(output_dir, f"article_{pmid}.xml")
            tree = ET.ElementTree(article)
            tree.write(filename, encoding="utf-8", xml_declaration=True)
            logging.info(f"Saved article {pmid} to {filename}")
        else:
            logging.warning(f"Could not find PMID for article {i+1}, skipping.")

def main():
    """Main function to run the research pipeline for both WNT and FGF."""
    for morphogen, term in SEARCH_TERMS.items():
        output_dir = f"{morphogen.lower()}_gradient_research"
        logging.info(f"--- Starting research for {morphogen} ---")
        try:
            id_list = search_pubmed(term, MAX_RESULTS)
            if id_list:
                article_data = fetch_articles(id_list)
                save_articles(article_data, output_dir)
                logging.info(f"Research for {morphogen} completed successfully.")
            else:
                logging.info(f"No articles found for {morphogen}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred during the request for {morphogen}: {e}")
        except ET.ParseError as e:
            logging.error(f"An error occurred while parsing XML for {morphogen}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred for {morphogen}: {e}")

if __name__ == "__main__":
    main()