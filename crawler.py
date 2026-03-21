import os
import re
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Set
from config import DATA_PATH

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; BahithBot/1.0; +https://github.com/SanaAraj/bahith-search)'
}

ARABIC_SOURCES = [
    # News
    ('https://www.aljazeera.net/news', 'الجزيرة'),
    ('https://www.bbc.com/arabic', 'بي بي سي عربي'),
    ('https://arabic.cnn.com', 'سي إن إن عربي'),
    # Tech
    ('https://aitnews.com', 'عالم التقنية'),
    # Science
    ('https://www.scientificamerican.com/arabic/', 'ساينتفك أمريكان'),
]


def extract_text(soup: BeautifulSoup) -> str:
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
        tag.decompose()

    text = soup.get_text(separator='\n', strip=True)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def has_arabic(text: str) -> bool:
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    arabic_chars = len(arabic_pattern.findall(text))
    return arabic_chars > 100


def get_page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find('h1')
    if h1:
        return h1.get_text(strip=True)
    return "untitled"


def crawl_page(url: str) -> dict:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or 'utf-8'

        soup = BeautifulSoup(resp.text, 'lxml')
        title = get_page_title(soup)
        text = extract_text(soup)

        if not has_arabic(text):
            return None

        if len(text) < 300:
            return None

        return {
            'url': url,
            'title': title,
            'content': text
        }
    except Exception as e:
        print(f"  Error crawling {url}: {e}")
        return None


def extract_links(soup: BeautifulSoup, base_url: str, domain: str) -> List[str]:
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if domain not in parsed.netloc:
            continue
        if parsed.scheme not in ('http', 'https'):
            continue
        if any(ext in parsed.path.lower() for ext in ['.jpg', '.png', '.gif', '.pdf', '.mp4', '.mp3']):
            continue

        links.append(full_url)
    return links


def crawl_site(start_url: str, source_name: str, max_pages: int = 10) -> List[dict]:
    parsed = urlparse(start_url)
    domain = parsed.netloc

    visited: Set[str] = set()
    to_visit = [start_url]
    pages = []

    print(f"\nCrawling {source_name} ({start_url})...")

    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)

        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = resp.apparent_encoding or 'utf-8'
            soup = BeautifulSoup(resp.text, 'lxml')

            new_links = extract_links(soup, url, domain)
            for link in new_links:
                if link not in visited:
                    to_visit.append(link)

            page = crawl_page(url)
            if page:
                page['source_name'] = source_name
                pages.append(page)
                print(f"  [{len(pages)}/{max_pages}] {page['title'][:50]}")

            time.sleep(1)

        except Exception as e:
            print(f"  Error: {e}")

    return pages


def save_crawled_pages(pages: List[dict]):
    os.makedirs(DATA_PATH, exist_ok=True)

    for page in pages:
        url_hash = hashlib.md5(page['url'].encode()).hexdigest()[:8]
        safe_title = re.sub(r'[^\w\s\u0600-\u06FF-]', '', page['title'])[:50].strip()
        filename = f"web_{url_hash}_{safe_title}.txt"
        filepath = os.path.join(DATA_PATH, filename)

        content = f"# {page['title']}\n"
        content += f"المصدر: {page['source_name']}\n"
        content += f"الرابط: {page['url']}\n\n"
        content += page['content']

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


def crawl_all(max_per_site: int = 5):
    all_pages = []

    for url, name in ARABIC_SOURCES:
        try:
            pages = crawl_site(url, name, max_pages=max_per_site)
            all_pages.extend(pages)
        except Exception as e:
            print(f"Failed to crawl {name}: {e}")

    if all_pages:
        save_crawled_pages(all_pages)
        print(f"\nSaved {len(all_pages)} pages to {DATA_PATH}/")

    return all_pages


def crawl_url(url: str, source_name: str = "ويب"):
    page = crawl_page(url)
    if page:
        page['source_name'] = source_name
        save_crawled_pages([page])
        print(f"Saved: {page['title']}")
        return page
    return None


if __name__ == "__main__":
    print("Arabic Web Crawler")
    print("=" * 50)

    pages = crawl_all(max_per_site=5)
    print(f"\nTotal pages crawled: {len(pages)}")
