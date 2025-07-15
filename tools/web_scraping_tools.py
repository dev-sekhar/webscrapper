# tools/web_scraping_tools.py

from langchain.tools import tool
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests


@tool
def fetch_static_content(url: str) -> str:
    """
    Fetches the clean text content from a single, static URL.
    Use this for simple websites that do not heavily rely on JavaScript.
    The input must be a valid URL string, like 'https://www.example.com'.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        text = " ".join(soup.body.get_text().split())
        return text[:4000]

    except requests.RequestException as e:
        return f"Error fetching URL {url}: {e}"


@tool
def browse_dynamic_page(url: str) -> str:
    """
    Opens a single, JavaScript-heavy webpage in a headless browser and returns its visible text content.
    Use this for dynamic websites like single-page applications.
    The input must be a valid URL string, like 'https://www.example.com'.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto(url, timeout=15000, wait_until="domcontentloaded")
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')

            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()

            visible_text = " ".join(soup.body.get_text().split())
            browser.close()
            return f"Successfully browsed {url}. Visible text: {visible_text[:4000]}"

        except Exception as e:
            browser.close()
            return f"Error during browser interaction at {url}: {e}"
