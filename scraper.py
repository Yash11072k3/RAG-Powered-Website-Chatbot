from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests


def scrape_dynamic(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        html = page.content()
        browser.close()
        return html


def scrape_static(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.text


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()

    noisy_selectors = [
        ".sidebar",
        ".toc",
        ".infobox",
        ".navbox",
        ".vector-header",
        ".vector-column-start",
        ".vector-page-toolbar",
        ".mw-editsection",
        ".mw-jump-link",
        ".reflist",
        ".reference",
        ".metadata",
        ".noprint",
        ".catlinks",
        ".printfooter",
        "#mw-navigation",
        "#footer",
        "#p-lang",
        "#p-search",
        "#toc",
    ]

    for selector in noisy_selectors:
        for el in soup.select(selector):
            el.decompose()

    preferred_selectors = [
        "article",
        "main",
        '[role="main"]',
        ".mw-parser-output",
        ".post-content",
        ".entry-content",
        ".article-content",
        "#content",
        ".content",
    ]

    main_content = None
    for selector in preferred_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break

    text = main_content.get_text(separator=" ", strip=True) if main_content else soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def scrape_website(url: str) -> str:
    try:
        try:
            html = scrape_dynamic(url)
        except Exception:
            html = scrape_static(url)

        text = clean_html(html)

        if not text or len(text.strip()) < 200:
            return "Error: Not enough readable content could be extracted from the website."

        return text

    except Exception as e:
        return f"Error: {str(e)}"