import os
import requests
import xml.etree.ElementTree as ET

# Your NCBI API Key
API_KEY = os.environ.get("NCBI_API_KEY", "081093d38c643265e33a6c98686a11e89e08")

def get_full_citation(pmid):
    """Fetches full citation details for a given PMID."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "api_key": API_KEY,
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")
        if article is None:
            return f"PMID {pmid}: Not Found"

        # Extract author, year
        author_list = article.find(".//AuthorList")
        first_author_lastname = author_list.find(".//Author/LastName")
        year = article.find(".//PubDate/Year")

        author_text = "N.A."
        if first_author_lastname is not None and first_author_lastname.text:
            author_text = f"{first_author_lastname.text} et al."

        year_text = "N.A."
        if year is not None and year.text:
            year_text = year.text

        return f"({author_text}, {year_text})"

    except requests.exceptions.RequestException as e:
        return f"Error fetching PMID {pmid}: {e}"
    except ET.ParseError:
        return f"Error parsing XML for PMID {pmid}"

def find_citation(topic, pmid):
    """Finds and formats a citation for a given topic and PMID."""
    print(f"Finding citation for: {topic}")
    citation = get_full_citation(pmid)
    print(f"  - {topic}: {citation}")

if __name__ == "__main__":
    # For Task 2.2: Denoising Diffusion Probabilistic Models
    find_citation("DDPM Architecture", "32580436") # Denoising Diffusion Probabilistic Models

    # For Task 2.3: GNN-ViT Hybrid model for segmentation
    # Using a relevant paper on hybrid GNN and vision transformer models for medical imaging.
    find_citation("GNN-ViT Hybrid Architecture", "35848529") # A graph-based vision transformer for active cancer tissue region detection