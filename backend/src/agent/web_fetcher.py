import io
import os
import re
from pathlib import Path
from typing import List, Optional, TypedDict, Union
from urllib.parse import urlparse

import httpx
import trafilatura
from dotenv import load_dotenv
from loguru import logger
from pypdf import PdfReader

load_dotenv()
blanks_pat = re.compile(r"[^\n\S]+")
dots_pat = re.compile(r"\.{3,}")
# Optional search providers
TAVILY_AVAILABLE = False
try:
    from tavily import TavilyClient

    if os.getenv("TAVILY_API_KEY"):
        TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False


class Doc(TypedDict):
    url: str
    title: str
    summary: str
    content: str


class SearchHit(TypedDict):
    query: str
    url: str
    title: str
    snippet: str


def clean_text(
    txt: str,
    max_chars: int = 50_000,
) -> str:
    txt = blanks_pat.sub(" ", txt).strip()
    txt = dots_pat.sub("...", txt)
    pre_max_chars = max_chars // 2
    post_max_chars = max_chars - pre_max_chars
    if len(txt) > (pre_max_chars + post_max_chars):
        txt = txt[:pre_max_chars] + " ... " + txt[-post_max_chars:]
    return txt


def parse_pdf_from_raw_text(
    raw_text: Union[str, bytes], max_chars: int = 50_000
) -> str:
    try:
        if isinstance(raw_text, str):
            raw_bytes = raw_text.encode("latin1", errors="ignore")
        else:
            raw_bytes = raw_text
        reader = PdfReader(io.BytesIO(raw_bytes))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                parts.append(text)
        return clean_text("\n".join(parts), max_chars=max_chars)
    except Exception as e:
        logger.error(f"parse_pdf_from_raw_text failed: {e}")
        return ""


def web_content_fetch(
    url: str, timeout: float = 25.0, max_chars: int = 50_000
) -> Optional[dict]:
    if not url or not url.startswith(("http://", "https://")):
        logger.warning(f"{url = } is not valid")
        return None
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "DeepResearchAgent/1.0"},
        ) as client:
            r = client.get(url)
            r.raise_for_status()
            downloaded = trafilatura.extract(
                r.text, include_comments=False, include_images=False, url=url
            )
            if not downloaded:
                logger.info(f"{url = } no content downloaded, using the raw text")
            content_type = r.headers.get("content-type", "").lower()
            is_pdf = url.endswith(".pdf") or "application/pdf" in content_type
            if is_pdf:
                logger.info(f"{url = } is a pdf file, parsing PDF bytes")
                content = parse_pdf_from_raw_text(r.content, max_chars)
            else:
                content = clean_text(
                    downloaded if downloaded else r.text, max_chars=max_chars
                )
            logger.info(f"{content[:200] = }")
            title_m = re.search(r"<title>(.*?)</title>", r.text, re.I)
            title = title_m.group(1).strip() if title_m else url
            return {"url": url, "title": title, "content": content}
    except Exception as e:
        logger.error(f"web_fetch failed for {url}: {e}")
        return None


def batch_web_contents_fetch(
    urls: List[str],
    timeout: float = 25.0,
    max_chars: int = 50_000,
    max_workers: int = 5,
) -> List[Optional[dict]]:
    """Fetch content from multiple URLs in parallel.

    Args:
        urls: List of URLs to fetch
        timeout: Timeout per request in seconds (default: 25.0)
        max_chars: Maximum content characters per fetch (default: 50_000)
        max_workers: Maximum number of parallel threads (default: 5)

    Returns:
        List of dicts with url, title, and content for successful fetches,
        or None for failed fetches. Order matches input urls order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Optional[dict]] = [None] * len(urls)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a mapping of future to index
        future_to_idx = {
            executor.submit(web_content_fetch, url, timeout, max_chars): idx
            for idx, url in enumerate(urls)
        }

        # Process completed futures and maintain order
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"batch_web_content_fetch error for URL index {idx}: {e}")
                results[idx] = None

    return results


def web_url_search(query: str, k: int = 6) -> List[SearchHit]:
    hits: List[SearchHit] = []
    if TAVILY_AVAILABLE:
        try:
            tv = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            res = tv.search(query=query, max_results=k)
            items = (
                res.get("results", [])
                if isinstance(res, dict)
                else (res if isinstance(res, list) else [])
            )
            for r in items:
                if isinstance(r, dict):
                    url = r.get("url") or ""
                    title = r.get("title") or url
                    snippet = (r.get("content") or r.get("snippet") or "")[:300]
                elif isinstance(r, str):
                    url = r if r.startswith(("http://", "https://")) else ""
                    title, snippet = (url or "Result"), ""
                else:
                    continue
                if url:
                    hits.append(
                        {"query": query, "url": url, "title": title, "snippet": snippet}
                    )
            return hits[:k]
        except Exception:
            pass

    return [
        {
            "query": query,
            "url": "",
            "title": "No search backend configured",
            "snippet": "Install tavily or ddg.",
        }
    ]


def _reg_domain(host: str) -> str:
    # Minimal registered-domain extractor: last two labels
    parts = (host or "").split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else (host or "")


def parse_url_parts(url: str):
    p = urlparse(url)
    scheme = (p.scheme or "").lower()
    host = p.hostname or ""
    regdom = _reg_domain(host)
    tld = regdom.split(".")[-1] if "." in regdom else regdom
    path = p.path or "/"
    query = p.query or ""
    return {
        "url": url,
        "scheme": scheme,
        "host": host,
        "regdom": regdom,
        "tld": tld,
        "path": path,
        "query": query,
    }


if __name__ == "__main__":
    logger.add(
        Path(__file__).with_suffix(".log"), mode="w", encoding="utf-8", level="DEBUG"
    )
    test_urls = [
        "https://pdf.dfcfw.com/pdf/H3_AP202503121644303318_1.pdf",
        "https://kenhuangus.substack.com/p/the-rise-of-agentic-ai-and-cyber",
        "https://mashable.com/article/agentic-ai-explainer",
    ]
    for url in test_urls:
        # result = web_fetch(url)
        result = parse_url_parts(url)
        if result:
            logger.info(f"result\n{result}")
        else:
            logger.error(f"Failed to fetch {url}")
