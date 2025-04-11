from newspaper import Article
from typing import Optional
from rich.console import Console
import concurrent.futures
import threading

console = Console()

def _download_article(article: Article) -> None:
    """Helper function to download article content in a separate thread."""
    article.download()

def extract_article_content(url: str, min_content_length: int = 500, timeout: int = 20) -> Optional[str]:
    """
    Extract the main content from a web article.
    
    Args:
        url: The URL of the article
        min_content_length: Minimum number of characters required for valid content
        timeout: Timeout in seconds for downloading and parsing the article
        
    Returns:
        The extracted article content if successful and meets length requirements, None otherwise
    """
    try:
        article = Article(url)
        
        # Download article with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_download_article, article)
            try:
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                console.print(f"[red]Timeout while downloading article from {url}[/red]")
                return None
                
        article.parse()
        
        content = article.text.strip()
        
        if len(content) < min_content_length:
            console.print(f"[yellow]Article content too short ({len(content)} chars) for URL: {url}[/yellow]")
            return None
            
        return content
    except Exception as e:
        console.print(f"[red]Error extracting content from {url}: {e}[/red]")
        return None 