# How to Use the Web Scraper Script

I'll guide you through using the web scraping script step by step:

## Prerequisites

First, make sure you have all the required dependencies installed:

```bash
pip install requests beautifulsoup4 pandas matplotlib numpy tqdm fake-useragent
```

## Basic Usage

The script requires at least one argument: the URL to scrape. Here's how to run it with just the required parameter:

```bash
python app.py --url https://example.com
```

This will:

1. Scrape up to 5 pages starting from example.com
2. Save results in an "output" directory
3. Use default settings for threads, rate limiting, etc.

## All Available Options

The script supports several command-line arguments to customize its behavior:

```bash
python app.py --url https://example.com --output my_results --pages 10 --threads 8 --timeout 15 --rate-limit 2.0 --no-cache
```

Here's what each parameter does:

 Parameter  Description  Default  `--url`  The base URL to start scraping from (required)  None  `--output`  Directory where results will be saved  "output"  `--pages`  Maximum number of pages to scrape  5  `--threads`  Number of concurrent threads for scraping  4  `--timeout`  Request timeout in seconds  10  `--rate-limit`  Time to wait between requests in seconds  1.0  `--no-cache`  Disable caching of requests  Cache enabled 

## Examples

### Basic scraping of a website (5 pages)

```bash
python app.py --url https://example.com
```

### Scrape more pages with faster crawling

```bash
python app.py --url https://example.com --pages 20 --threads 8
```

### Be more respectful to the server (slower requests)

```bash
python app.py --url https://example.com --rate-limit 3.0
```

### Disable caching for fresh results

```bash
python app.py --url https://example.com --no-cache
```

## Output Files

After running the script, you'll find the following files in the output directory:

1. `scraping_results.json` - Raw scraped data in JSON format
2. `scraping_results.csv` - Flattened data in CSV format
3. `scraping_analysis.png` - Visualizations of the scraped data
