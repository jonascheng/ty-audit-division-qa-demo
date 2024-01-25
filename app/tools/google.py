import logging

from langchain_community.utilities import GoogleSearchAPIWrapper


# Get logger
logger = logging.getLogger(__name__)


def google_search(query, site_name, site_link=None) -> list[dict]:
    """
    When user asks you to "search on google" always use this tool.

    Input is the query and an optional site.
    """
    # instantiate a Google Search API Wrapper object for search capability
    search = GoogleSearchAPIWrapper()

    if not query:
        return "Please provide a query."

    if site_name:
        query = f'{site_name} {query}'

    # the data structure of search_results is
    # a list of SearchResult objects
    # each SearchResult object has the following attributes:
    # title, link, snippet
    search_results = search.results(
        query,
        num_results=10)

    result_set = []
    for i, result in enumerate(search_results):
        link = result.get('link')

        # st.write(result)

        # if site_link is provided, only keep results from that site
        if site_link and site_link not in link:
            continue

        result_set.append(result)

    return result_set
