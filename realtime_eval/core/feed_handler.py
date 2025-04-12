import json
import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

date_formats = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S GMT"
]

def load_feeds() -> List[Dict]:
    """Load RSS feed URLs from the JSON file."""
    try:
        with open('feeds.json', 'r') as f:
            return json.load(f)['feeds']
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[red]Error loading feeds: {str(e)}[/red]")
        return []

def fetch_feed(feed_url: str) -> feedparser.FeedParserDict:
    """Fetch and parse an RSS feed."""
    try:
        response = requests.get(feed_url, timeout=10)
        response.raise_for_status()
        return feedparser.parse(response.content)
    except (requests.RequestException, requests.Timeout) as e:
        console.print(f"[red]Error fetching feed {feed_url}: {e}[/red]")
        return feedparser.FeedParserDict()

def format_date(date_str: str) -> str:
    """Format the date string to a more readable format."""    
    for date_format in date_formats:
        try:
            date = datetime.strptime(date_str, date_format)
            return date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    raise ValueError(f"No valid date format found for {date_str}")

def is_within_24_hours(date_str: str) -> bool:
    """Check if the date is within the last 24 hours."""

    for date_format in date_formats:
        try:
            date = datetime.strptime(date_str, date_format)
            now = datetime.now(date.tzinfo) if date.tzinfo else datetime.utcnow()
            return date > now - timedelta(days=1)
        except ValueError:
            continue
    return False

def is_within_7_days(date_str: str) -> bool:
    """Check if the date is within the last 7 days."""

    for date_format in date_formats:
        try:
            date = datetime.strptime(date_str, date_format)
            now = datetime.now(date.tzinfo) if date.tzinfo else datetime.utcnow()
            return date > now - timedelta(days=7)
        except ValueError:
            continue
    return False
    
def display_articles(feed: feedparser.FeedParserDict, feed_name: str):
    """Display articles from a feed in a rich table."""
    if not feed.entries:
        return

    table = Table(title=f"{feed_name} - {feed.feed.get('title', 'Unknown')}")
    table.add_column("Title", style="cyan")
    table.add_column("Published", style="green")
    table.add_column("Link", style="blue")

    for entry in feed.entries:
        title = entry.get('title', 'No title')
        link = entry.get('link', 'No link')
        published = format_date(entry.get('published', 'No date'))
        table.add_row(title, published, link)

    console.print(Panel(table, title=feed_name, border_style="yellow")) 