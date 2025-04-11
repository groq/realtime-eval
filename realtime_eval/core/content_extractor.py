from newspaper import Article
from typing import Optional
from rich.console import Console

console = Console()

def extract_article_content(url: str, min_content_length: int = 500) -> Optional[str]:
    """
    Extract the main content from a web article.
    
    Args:
        url: The URL of the article
        min_content_length: Minimum number of characters required for valid content
        
    Returns:
        The extracted article content if successful and meets length requirements, None otherwise
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        content = article.text.strip()
        
        if len(content) < min_content_length:
            console.print(f"[yellow]Article content too short ({len(content)} chars) for URL: {url}[/yellow]")
            return None
            
        return content
    except Exception as e:
        console.print(f"[red]Error extracting content from {url}: {e}[/red]")
        return None 