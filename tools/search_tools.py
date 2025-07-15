# tools/search_tools.py

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
# We need to import the 'Tool' class to wrap the older utility
from langchain.tools import Tool

# --- Tool 1: Tavily (The AI-Optimized Researcher) ---
# This tool is modern and already a valid BaseTool instance.
tavily_search = TavilySearchResults(
    max_results=5,
    api_key=os.environ.get("TAVILY_API_KEY")
)
tavily_search.name = "tavily_search_engine"
tavily_search.description = """
A powerful search engine optimized for AI agents. 
Useful for deep research, finding articles, and when you need the full text content of web pages.
"""

# --- Tool 2: SerpAPI (The Google Scraper) ---
# This is the corrected way to handle older utilities.

# 1. First, create an instance of the utility wrapper.
serpapi_utility = SerpAPIWrapper(
    serpapi_api_key=os.environ.get("SERPAPI_API_KEY"))

# 2. Then, wrap it in the 'Tool' class to make it agent-ready.
#    This gives it the .name, .description, and .func attributes the agent needs.
google_serp_search = Tool(
    name="google_serp_search",
    description="""
A tool that scrapes Google search results.
Use this when you need Google-style results with rich metadata.
Especially useful for finding specific SERP features like videos, maps, shopping results, or the 'People also ask' section.
""",
    func=serpapi_utility.run,  # The function the agent will call
)

# You can still comment out the Brave Search tool if you are not using it.
# # --- Tool 3: Brave (The Fast & Factual Engine) ---
# from langchain_brave_search.tools import BraveSearch
# brave_search = BraveSearch.from_api_key(
#     api_key=os.environ.get("BRAVE_API_KEY"),
#     search_kwargs={"count": 5}
# )
# brave_search.name = "brave_fast_search"
# brave_search.description = """
# An independent search engine for fast, factual answers and current events.
# """

# The tool_loader will now discover 'tavily_search' and 'google_serp_search'
# as valid BaseTool instances.
