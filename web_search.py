import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from preprocessor import preprocess

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """Search DuckDuckGo for Arabic content"""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": "xa-ar"}  # Arabic region

    try:
        resp = requests.post(url, data=params, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'lxml')

        results = []
        for r in soup.select('.result')[:max_results]:
            title_elem = r.select_one('.result__title')
            snippet_elem = r.select_one('.result__snippet')
            link_elem = r.select_one('.result__url')

            if title_elem and snippet_elem:
                url = link_elem.get_text(strip=True) if link_elem else ''
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'snippet': snippet_elem.get_text(strip=True),
                    'url': url,
                    'source': url if url else 'web'
                })

        return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def fetch_page_content(url: str) -> str:
    """Fetch and extract text from a URL"""
    try:
        if not url.startswith('http'):
            url = 'https://' + url

        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = resp.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')

        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return preprocess(text[:2000])
    except:
        return ""


def live_web_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search the web and return results with content"""
    results = search_duckduckgo(query, max_results)

    for r in results:
        if r.get('url'):
            content = fetch_page_content(r['url'])
            if content:
                r['content'] = content
            else:
                r['content'] = r['snippet']
        else:
            r['content'] = r['snippet']

        r['score'] = 0.8  # Web results get a base score

    return results


if __name__ == "__main__":
    query = "الذكاء الاصطناعي في السعودية"
    print(f"Searching web for: {query}\n")

    results = live_web_search(query, max_results=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        print(f"   URL: {r['url']}")
        print(f"   {r['snippet'][:100]}...")
        print()
