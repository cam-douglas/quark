import arxiv
import sys

def research_csf(task_description):
    """
    Performs a research task using the arxiv library and prints a summary.
    """
    try:
        # Perform a search on arXiv
        search = arxiv.Search(
            query=task_description,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        # Process the search results
        summary = ""
        for result in search.results():
            summary += f"Title: {result.title}\n"
            summary += f"Authors: {', '.join(author.name for author in result.authors)}\n"
            summary += f"URL: {result.entry_id}\n"
            summary += f"Abstract: {result.summary}\n\n"

        print(summary)

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    task = "Research the physical properties (e.g., viscosity, density, pressure) and flow dynamics of cerebrospinal fluid in the early embryonic ventricular system. Also, research its role as a medium for morphogen transport. Reference at least 3 peer-reviewed sources."
    research_csf(task)