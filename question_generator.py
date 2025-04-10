import json
import feedparser
import requests
from datetime import datetime
from typing import List, Dict, Optional
from rich.console import Console
from rich.progress import Progress
from groq import Groq
import os
from dataclasses import dataclass

console = Console()

@dataclass
class Article:
    title: str
    link: str
    date: str
    question: Optional[str] = None
    answer: Optional[str] = None

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

def generate_question_and_answer(title: str, client: Groq) -> Optional[Dict[str, str]]:
    """Generate a question and answer based on the article title using Groq API."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates questions and answers in JSON format to test an LLM's ability to access real-time information from news headlines. "
                        "Use the following guidelines:\n\n"
                        "1. Analyze the news headline and determine if it contains enough clear, specific details to generate a precise question and a factual answer. "
                        "The headline must refer to a real-time event, include a date, or be specific enough to remain relevant beyond a few weeks. "
                        "If the headline is too vague or does not contain a clear fact, output the string 'SKIP' in both question and answer values.\n\n"
                        "2. When generating the question, make it very specific and include the date or any time-related detail from the headline. "
                        "For example, instead of asking a general question like 'What happened with the stock market?', include something like "
                        "'What happened in the stock market on March 10, 2025, following the tariff pause announcement?'\n\n"
                        "3. The answer should directly summarize the core fact(s) from the headline, ensuring the answer is clearly supported by the headline details.\n\n"
                        "4. Your response must be in a JSON schema with two keys: 'question' and 'answer', both of which should have string values.\n\n"
                        "5. Provide many examples of good questions and answers based on various headlines."
                    )
                },
                {
                    "role": "user",
                    "content": f"Generate a question and answer based on this news headline: {title}"
                }
            ],
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            console.print(f"[red]Empty response for headline: {title}[/red]")
            return None
        result = json.loads(content)
        if result.get("question") == "SKIP" or result.get("answer") == "SKIP":
            return None
        return result
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON response for headline: {title}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error generating question and answer: {e}[/red]")
        return None

def process_articles(feeds: List[Dict], client: Groq, test: bool = False) -> List[Article]:
    """Process articles from all feeds and generate questions and answers."""
    articles = []
    max_articles = 5 if test else None
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing articles...", total=len(feeds))
        
        for feed_info in feeds:
            feed = fetch_feed(feed_info['url'])
            if not feed.entries:
                continue
                
            for i, entry in enumerate(feed.entries):
                if max_articles and i >= max_articles:
                    break
                title = entry.get('title', 'No title')
                link = entry.get('link', 'No link')
                date = format_date(entry.get('published', 'No date'))
                
                qa_pair = generate_question_and_answer(title, client)
                if qa_pair:
                    articles.append(Article(
                        title=title,
                        link=link,
                        date=date,
                        question=qa_pair['question'],
                        answer=qa_pair['answer']
                    ))
            
            progress.update(task, advance=1)
    
    return articles

def save_dataset(articles: List[Article], filename: str = "news_questions.json"):
    """Save the generated questions and answers to a JSON file."""
    dataset = [
        {
            "question": article.question,
            "answer": article.answer,
            "title": article.title,
            "link": article.link,
            "date": article.date
        }
        for article in articles
    ]
    
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    console.print(f"[green]Dataset saved to {filename} with {len(articles)} entries[/green]")

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