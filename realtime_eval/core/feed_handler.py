import json
import feedparser
import requests
from datetime import datetime
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

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
        response = requests.get(feed_url)
        response.raise_for_status()
        return feedparser.parse(response.content)
    except requests.RequestException as e:
        console.print(f"[red]Error fetching feed {feed_url}: {e}[/red]")
        return feedparser.FeedParserDict()

def format_date(date_str: str) -> str:
    """Format the date string to a more readable format."""
    try:
        date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
        return date.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return date_str

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