import argparse
import csv
import datetime
import functools
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
import concurrent.futures
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages using: pip install requests beautifulsoup4 pandas matplotlib numpy tqdm fake-useragent")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("WebScraper")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]

DB_CACHE_PATH = "request_cache.db"
cache_lock = Lock()


@dataclass
class ScraperConfig:
    base_url: str
    output_dir: str = "output"
    max_pages: int = 5
    max_threads: int = 4
    request_timeout: int = 10
    rate_limit: float = 1.0   
    use_cache: bool = True
    cache_expiry: int = 86400 
    proxies: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    user_agent_rotation: bool = True
    follow_redirects: bool = True
    parse_javascript: bool = False
    data_selectors: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        #default headers if none provided
        if not self.headers:
            self.headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }


class RequestCache:
    
    def __init__(self, db_path: str = DB_CACHE_PATH, expiry: int = 86400):
        self.db_path = db_path
        self.expiry = expiry
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    url TEXT PRIMARY KEY,
                    response TEXT,
                    content_type TEXT,
                    status_code INTEGER,
                    timestamp INTEGER
                )
            ''')
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        with cache_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response, content_type, status_code, timestamp FROM cache WHERE url = ?",
                (url,)
            )
            result = cursor.fetchone()
            
            if result:
                response_text, content_type, status_code, timestamp = result
                current_time = int(time.time())
                
                if current_time - timestamp > self.expiry:
                    return None
                
                return {
                    "response": response_text,
                    "content_type": content_type,
                    "status_code": status_code,
                    "cached": True,
                    "timestamp": timestamp
                }
            
            return None
    
    def set(self, url: str, response_text: str, content_type: str, status_code: int):
        """Store a response in the cache."""
        current_time = int(time.time())
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)",
                (url, response_text, content_type, status_code, current_time)
            )
            conn.commit()
    
    def clear(self, url: Optional[str] = None):
        """Clear the cache for a specific URL or all URLs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if url:
                cursor.execute("DELETE FROM cache WHERE url = ?", (url,))
            else:
                cursor.execute("DELETE FROM cache")
            conn.commit()
    
    def clear_expired(self):
        """Clear expired cache entries."""
        current_time = int(time.time())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache WHERE ? - timestamp > ?",
                (current_time, self.expiry)
            )
            conn.commit()


class RateLimiter:
    """Rate limiter for HTTP requests to avoid overwhelming servers."""
    
    def __init__(self, rate_limit: float = 1.0):
        """Initialize with rate limit in seconds."""
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.lock = Lock()
    
    def wait(self):
        """Wait until rate limit allows next request."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.rate_limit:
                sleep_time = self.rate_limit - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


def get_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """Create a requests session with retry capability."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_random_user_agent() -> str:
    """Get a random user agent string."""
    try:
        ua = UserAgent()
        return ua.random
    except:
        return random.choice(USER_AGENTS)


def fetch_url(
    url: str, 
    config: ScraperConfig,
    rate_limiter: RateLimiter,
    cache: RequestCache
) -> Dict[str, Any]:
    """
    Fetch a URL with caching, rate limiting, and proxy rotation.
    
    Args:
        url: URL to fetch
        config: Scraper configuration
        rate_limiter: Rate limiter instance
        cache: Request cache instance
        
    Returns:
        Dictionary with response data
    """
    if config.use_cache:
        cached_response = cache.get(url)
        if cached_response:
            logger.debug(f"Cache hit for {url}")
            return cached_response
    
    rate_limiter.wait()
    session = get_session()
    headers = config.headers.copy()
    
    if config.user_agent_rotation:
        headers["User-Agent"] = get_random_user_agent()
    
    proxies = None
    if config.proxies:
        proxy = random.choice(config.proxies)
        proxies = {"http": proxy, "https": proxy}
    
    try:
        response = session.get(
            url,
            headers=headers,
            proxies=proxies,
            timeout=config.request_timeout,
            allow_redirects=config.follow_redirects
        )
        
        response.raise_for_status()
        
        result = {
            "response": response.text,
            "content_type": response.headers.get("Content-Type", ""),
            "status_code": response.status_code,
            "cached": False,
            "timestamp": int(time.time())
        }
        
        if config.use_cache:
            cache.set(
                url, 
                response.text, 
                response.headers.get("Content-Type", ""), 
                response.status_code
            )
        
        return result
    
    except requests.RequestException as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return {
            "error": str(e),
            "status_code": getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0,
            "cached": False
        }


def parse_html(html_content: str, url: str) -> BeautifulSoup:
    """Parse HTML content with BeautifulSoup."""
    return BeautifulSoup(html_content, "html.parser")


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """Extract all links from a BeautifulSoup object."""
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)

        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.append(full_url)
    
    return links


def extract_data(soup: BeautifulSoup, selectors: Dict[str, str]) -> Dict[str, Any]:
    """Extract data from a BeautifulSoup object using CSS selectors."""
    data = {}
    
    for key, selector in selectors.items():
        elements = soup.select(selector)
        
        if not elements:
            data[key] = None
            continue
        if key.endswith("_list"):
            # Extract a list of text values
            data[key] = [el.get_text(strip=True) for el in elements]
        elif key.endswith("_href"):
            # Extract href attributes
            data[key] = [el.get("href") for el in elements if el.get("href")]
        elif key.endswith("_src"):
            # Extract src attributes
            data[key] = [el.get("src") for el in elements if el.get("src")]
        elif key.endswith("_html"):
            # Extract raw HTML
            data[key] = [str(el) for el in elements]
        elif key.endswith("_attr"):
            # Extract all attributes as a dict
            attr_name = key.rsplit("_", 2)[1]  # Extract attribute name from key
            data[key] = [el.get(attr_name) for el in elements if el.get(attr_name)]
        else:
            # Default to extracting text from the first matching element
            data[key] = elements[0].get_text(strip=True)
    
    return data


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace special characters with their plain equivalents
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&nbsp;', ' ')
    
    return text


def scrape_page(url: str, config: ScraperConfig, rate_limiter: RateLimiter, cache: RequestCache) -> Dict[str, Any]:
    """
    Scrape a single page and extract data.
    
    Args:
        url: URL to scrape
        config: Scraper configuration
        rate_limiter: Rate limiter instance
        cache: Request cache instance
        
    Returns:
        Dictionary with extracted data
    """
    logger.info(f"Scraping {url}")
    
    # Fetch the URL
    response_data = fetch_url(url, config, rate_limiter, cache)
    
    if "error" in response_data:
        logger.error(f"Failed to fetch {url}: {response_data['error']}")
        return {"url": url, "success": False, "error": response_data["error"]}
    
    # Parse the HTML
    html_content = response_data["response"]
    soup = parse_html(html_content, url)
    
    # Extract links
    links = extract_links(soup, config.base_url)
    
    # Extract data using selectors
    data = extract_data(soup, config.data_selectors)
    
    # Clean extracted text data
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = clean_text(value)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            data[key] = [clean_text(item) for item in value]
    
    # Add metadata
    result = {
        "url": url,
        "success": True,
        "title": soup.title.string if soup.title else "",
        "links": links,
        "timestamp": datetime.datetime.now().isoformat(),
        "data": data
    }
    
    return result


def crawl_website(config: ScraperConfig) -> List[Dict[str, Any]]:
    """
    Crawl a website starting from the base URL.
    
    Args:
        config: Scraper configuration
        
    Returns:
        List of dictionaries with scraped data
    """
    # Initialize resources
    rate_limiter = RateLimiter(config.rate_limit)
    cache = RequestCache(expiry=config.cache_expiry)
    
    # Clear expired cache entries
    cache.clear_expired()
    
    # Queue of URLs to scrape
    to_visit = [config.base_url]
    visited = set()
    results = []
    
    with tqdm(total=config.max_pages, desc="Scraping pages") as pbar:
        with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
            while to_visit and len(visited) < config.max_pages:
                # Get next batch of URLs to process
                current_batch = []
                while to_visit and len(current_batch) < config.max_threads:
                    url = to_visit.pop(0)
                    if url not in visited:
                        visited.add(url)
                        current_batch.append(url)
                
                if not current_batch:
                    break
                
                # Submit batch of URLs to thread pool
                future_to_url = {
                    executor.submit(scrape_page, url, config, rate_limiter, cache): url
                    for url in current_batch
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Add new links to the queue
                        if result["success"]:
                            new_links = [
                                link for link in result.get("links", [])
                                if link not in visited and link not in to_visit
                            ]
                            to_visit.extend(new_links[:config.max_pages - len(visited)])
                        
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
                        results.append({
                            "url": url,
                            "success": False,
                            "error": str(e)
                        })
                        pbar.update(1)
    
    logger.info(f"Crawling complete. Scraped {len(results)} pages.")
    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the scraped data.
    
    Args:
        results: List of scraped data dictionaries
        
    Returns:
        Dictionary with analysis results
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            "url": r["url"],
            "success": r["success"],
            "title": r.get("title", ""),
            "num_links": len(r.get("links", [])),
            "timestamp": r.get("timestamp", ""),
            **r.get("data", {})
        }
        for r in results if r["success"]
    ])
    
    if df.empty:
        return {"error": "No successful results to analyze"}
    
    # Basic statistics
    analysis = {
        "total_pages": len(results),
        "successful_pages": df["success"].sum(),
        "failed_pages": len(results) - df["success"].sum(),
        "average_links_per_page": df["num_links"].mean(),
        "most_linked_urls": Counter([
            link for r in results if r["success"] 
            for link in r.get("links", [])
        ]).most_common(10)
    }
    
    # Analyze text content if available
    text_columns = [col for col in df.columns if isinstance(df[col].iloc[0], str) if df[col].iloc[0]]
    if text_columns:
        # Calculate average text length
        for col in text_columns:
            analysis[f"avg_{col}_length"] = df[col].str.len().mean()
        
        # Word frequency analysis on combined text
        all_text = " ".join(df[text_columns[0]].dropna())
        words = re.findall(r'\b\w+\b', all_text.lower())
        analysis["word_frequency"] = Counter(words).most_common(20)
    
    return analysis


def visualize_results(results: List[Dict[str, Any]], analysis: Dict[str, Any], output_dir: str):
    """
    Create visualizations from the scraped data.
    
    Args:
        results: List of scraped data dictionaries
        analysis: Analysis results dictionary
        output_dir: Directory to save visualizations
    """
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Web Scraping Results Analysis", fontsize=16)
    
    # Plot 1: Success vs. Failure pie chart
    labels = ['Success', 'Failure']
    sizes = [analysis["successful_pages"], analysis["failed_pages"]]
    axs[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axs[0, 0].set_title('Scraping Success Rate')
    
    # Plot 2: Top 10 most linked URLs
    if analysis.get("most_linked_urls"):
        urls = [url[:30] + "..." if len(url) > 30 else url for url, _ in analysis["most_linked_urls"]]
        counts = [count for _, count in analysis["most_linked_urls"]]
        axs[0, 1].barh(urls, counts)
        axs[0, 1].set_title('Top Linked URLs')
        axs[0, 1].set_xlabel('Count')
    
    # Plot 3: Links per page histogram
    if "num_links" in pd.DataFrame([r for r in results if r["success"]]).columns:
        df = pd.DataFrame([r for r in results if r["success"]])
        axs[1, 0].hist(df["num_links"], bins=20)
        axs[1, 0].set_title('Links per Page Distribution')
        axs[1, 0].set_xlabel('Number of Links')
        axs[1, 0].set_ylabel('Frequency')
    
    # Plot 4: Word frequency bar chart
    if analysis.get("word_frequency"):
        words = [word for word, _ in analysis["word_frequency"][:10]]
        freqs = [freq for _, freq in analysis["word_frequency"][:10]]
        axs[1, 1].barh(words, freqs)
        axs[1, 1].set_title('Most Frequent Words')
        axs[1, 1].set_xlabel('Frequency')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "scraping_analysis.png"))
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def export_results(results: List[Dict[str, Any]], output_dir: str):
    """
    Export results to multiple formats.
    
    Args:
        results: List of scraped data dictionaries
        output_dir: Directory to save exports
    """
    # Export to JSON
    json_path = os.path.join(output_dir, "scraping_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Export to CSV (flattening the data structure)
    csv_path = os.path.join(output_dir, "scraping_results.csv")
    
    # Convert to DataFrame with flattened structure
    flat_data = []
    for result in results:
        if result["success"]:
            row = {
                "url": result["url"],
                "title": result.get("title", ""),
                "timestamp": result.get("timestamp", "")
            }
            
            # Flatten the data dictionary
            for key, value in result.get("data", {}).items():
                if isinstance(value, list):
                    row[key] = ", ".join(str(v) for v in value)
                else:
                    row[key] = value
            
            flat_data.append(row)
    
    if flat_data:
        pd.DataFrame(flat_data).to_csv(csv_path, index=False)
    
    logger.info(f"Results exported to {output_dir}")


def main():
    """Main function to run the web scraper."""
    parser = argparse.ArgumentParser(description="Advanced Web Scraper")
    parser.add_argument("--url", type=str, required=True, help="Base URL to scrape")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--pages", type=int, default=5, help="Maximum pages to scrape")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Rate limit in seconds")
    parser.add_argument("--no-cache", action="store_true", help="Disable request caching")
    args = parser.parse_args()
    
    # Example selectors for common data - these would be customized for the target site
    data_selectors = {
        "heading": "h1",
        "subheadings_list": "h2, h3",
        "main_content": "article p, main p, .content p",
        "image_urls_src": "img",
        "links_href": "a",
        "prices_list": ".price, [data-price]",
        "metadata": "meta[name='description']",
        "author": ".author, [rel='author']",
        "date": ".date, time, [datetime]",
    }
    
    # Create configuration
    config = ScraperConfig(
        base_url=args.url,
        output_dir=args.output,
        max_pages=args.pages,
        max_threads=args.threads,
        request_timeout=args.timeout,
        rate_limit=args.rate_limit,
        use_cache=not args.no_cache,
        data_selectors=data_selectors
    )
    
    # Run the scraper
    try:
        logger.info(f"Starting web scraping of {config.base_url}")
        results = crawl_website(config)
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Visualize results
        visualize_results(results, analysis, config.output_dir)
        
        # Export results
        export_results(results, config.output_dir)
        
        logger.info("Web scraping completed successfully")
    except Exception as e:
        logger.error(f"Error in web scraping: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())