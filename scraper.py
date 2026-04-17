from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urldefrag
import requests


def normalize_url(url: str) -> str:
    clean_url, _ = urldefrag(url.strip())
    return clean_url


def scrape_dynamic(url: str) -> str:
    url = normalize_url(url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900}
        )
        page = context.new_page()

        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)

        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        html = page.content()
        browser.close()
        return html


def scrape_static(url: str) -> str:
    url = normalize_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    response = requests.get(url, headers=headers, timeout=25)
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
        ".cookie",
        ".popup",
        ".modal",
        ".newsletter",
    ]

    for selector in noisy_selectors:
        for el in soup.select(selector):
            el.decompose()

    preferred_selectors = [
        "main",
        "article",
        '[role="main"]',
        ".mw-parser-output",
        ".content",
        "#content",
        ".page-content",
        ".post-content",
        ".entry-content",
    ]

    main_content = None
    for selector in preferred_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break

    text = main_content.get_text(separator=" ", strip=True) if main_content else soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def scrape_website(url: str) -> str:
    url = normalize_url(url)

    try:
        try:
            html = scrape_dynamic(url)
        except Exception:
            html = scrape_static(url)

        text = clean_html(html)

        if not text or len(text.strip()) < 200:
            return "Error: Not enough readable content could be extracted from the website."

        return text

    except requests.exceptions.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code == 403:
            return "Error: This website is blocking automated access (403 Forbidden). Try another public page or use a site that allows scraping."
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"