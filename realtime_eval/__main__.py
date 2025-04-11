import os
from rich.console import Console
from groq import Groq
from .core.feed_handler import load_feeds
from .core.question_generator import process_articles, save_dataset

console = Console()

def main(test: bool = False):
    # Check for Groq API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print("[red]Error: GROQ_API_KEY environment variable not set[/red]")
        return
    
    client = Groq(api_key=api_key)
    feeds = load_feeds()
    
    if not feeds:
        return
    
    articles = process_articles(feeds, client, test)
    save_dataset(articles)

if __name__ == "__main__":
    main(test=True)  # Set test=True to process only the first 5 articles 