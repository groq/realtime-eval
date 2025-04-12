import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress
from groq import Groq
from .feed_handler import fetch_feed, format_date, load_feeds, is_within_24_hours
from .content_extractor import extract_article_content

console = Console()

@dataclass
class Article:
    title: str
    link: str
    date: str
    content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    answer_context: Optional[str] = None

def generate_questions_and_answers(title: str, content: str, client: Groq) -> Optional[List[Dict[str, str]]]:
    """Generate up to 3 questions and answers based on the article content using Groq API."""
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates questions and answers in JSON format to test an LLM's ability to access real-time information from news articles. "
                        "Use the following guidelines:\n\n"
                        "1. Analyze the article content and generate up to 3 specific questions that can be answered using direct quotes or specific information from the article. "
                        "* Each question should be about something that has happened or been learned only in the past 24 hours."
                        "* Each question should ask a question that can be researched rather than referencing the specific article, as the answerer will be searching for the answer online instead of having the specific article to reference."
                        "* Each question should be clear, specific, and test the ability to find information within the text.\n\n"
                        "2. Each answer should be a direct quote or specific information from the article that answers the question. "
                        "Include the exact text from the article that contains the answer.\n\n"
                        "3. Your response must be in a JSON schema with an array of objects, each containing: 'question', 'answer', and 'answer_context'. "
                        "'answer_context' should contain the exact text from the article that contains the answer.\n\n"
                        "4. If the article doesn't contain enough specific information to generate good question-answer pairs, output 'SKIP' for all values.\n\n"
                        "Example response:\n"
                        "{\n"
                        "  \"qa_pairs\": [\n"
                        "    {\n"
                        "      \"question\": \"What specific action did the Federal Reserve announce regarding interest rates?\",\n"
                        "      \"answer\": \"The Federal Reserve announced it would maintain the current interest rates.\",\n"
                        "      \"answer_context\": \"In a statement released today, the Federal Reserve announced it would maintain the current interest rates, citing stable economic indicators.\"\n"
                        "    },\n"
                        "    {\n"
                        "      \"question\": \"What was the reported inflation rate for October 2024 that influenced the Fed's decision to maintain interest rates?\",\n"
                        "      \"answer\": \"The inflation rate was 3.2% in October 2024\",\n"
                        "      \"answer_context\": \"The Federal Reserve's decision was influenced by the latest economic data showing inflation at 3.2% in October 2024, down from 3.7% in September 2024.\"\n"
                        "    }\n"
                        "  ]\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Article title: {title}\n\nArticle content:\n{content}\n\nGenerate up to 3 questions and answers based on this article:"
                }
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            console.print(f"[red]Empty response for article: {title}[/red]")
            return None
        result = json.loads(content)
        qa_pairs = result.get("qa_pairs", [])
        if not qa_pairs or qa_pairs[0].get("question") == "SKIP":
            return None
        return qa_pairs
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON response for article: {title}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error generating questions and answers: {e}[/red]")
        return None

def evaluate_questions(articles: List[Article], client: Groq) -> List[int]:
    """Evaluate questions and answers using an LLM to determine which to keep."""
    indices_to_keep = []
    offset = 0
    console.print(f"[blue]Total number of articles: {len(articles)}[/blue]")  # Debug log
    
    for i in range(0, len(articles), 5):
        batch = articles[i:i+5]
        console.print(f"[blue]Processing batch starting at index {i}, batch size: {len(batch)}[/blue]")  # Debug log
        batch_content = [
            {
                "question": article.question,
                "answer": article.answer
            }
            for article in batch
        ]
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that evaluates questions and answers in JSON format to test an LLM's ability to access real-time information from news headlines. "
                            "Use the following guidelines:\n\n"
                            "1. Analyze each question and answer pair to determine if it meets the criteria of being clear, specific, and based on a real-time event. The question should include a date or time-related detail from the article, or be specific enough to remain relevant beyond a few weeks. The question MUST be about something that has happened or been learned only in the past 24 hours."
                            "If the pair is strong, include its index in the response.\n\n"
                            "2. Return a JSON object with reasoning and indices fields."
                            "Example response: {\"reasoning\": \"Pair 0 is good because...\", \"indices\": [0, 1, 2]}"
                            "Example pairs:\n"
                            "[{\"question\": \"What was the outcome of SpaceX's launch yesterday?\", \"answer\": \"SpaceX successfully launched 22 Starlink satellites yesterday from Florida.\"},\n"
                            " {\"question\": \"Who won the 2023 World Series?\", \"answer\": \"The Texas Rangers won the 2023 World Series.\"},\n"
                            " {\"question\": \"What did the Fed announce about interest rates this week?\", \"answer\": \"The Federal Reserve announced it is maintaining current interest rates this week.\"}]"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Evaluate these question-answer pairs: {json.dumps(batch_content)}"
                    }
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            print(response.choices[0].message.content)
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            new_indices_to_keep = result.get('indices', [])
            console.print(f"[blue]Received indices from LLM: {new_indices_to_keep}, current offset: {offset}[/blue]")  # Debug log
            indices_to_keep.extend([i + offset for i in new_indices_to_keep])
            offset += 5

            console.print(f"[green]Running indices to keep: {indices_to_keep}[/green]")
        except Exception as e:
            console.print(f"[red]Error evaluating questions: {e}[/red]")
            return []
    
    console.print(f"[blue]Final indices to keep: {indices_to_keep}[/blue]")  # Debug log
    return indices_to_keep

def process_articles(feeds: List[Dict], client: Groq, test: bool = False, timeout: int = 20) -> List[Article]:
    """Process articles from all feeds and generate questions and answers.
    
    Args:
        feeds: List of feed dictionaries containing URLs
        client: Groq client instance
        test: Whether to run in test mode (limited articles)
        timeout: Timeout in seconds for processing each article
    """
    articles = []
    max_articles = 5 if test else None
    
    with Progress() as progress:
        feed_task = progress.add_task("[cyan]Processing feeds...", total=len(feeds))
        
        for feed_info in feeds:
            feed = fetch_feed(feed_info['url'])
            if not feed.entries:
                progress.update(feed_task, advance=1)
                continue
            
            # Filter out articles that are older than 24 hours
            feed.entries = [entry for entry in feed.entries if is_within_24_hours(entry.get('published', 'No date'))]
            
            # Calculate total articles to process for this feed
            total_articles = min(len(feed.entries), max_articles) if max_articles else len(feed.entries)
            article_task = progress.add_task(f"[cyan]Processing articles from {feed_info['url']}...", total=total_articles)
            
            for i, entry in enumerate(feed.entries):
                if max_articles and i >= max_articles:
                    break
                    
                title = entry.get('title', 'No title')
                link = entry.get('link', 'No link')
                date = format_date(entry.get('published', 'No date'))
                
                try:
                    # Extract article content with timeout
                    content = extract_article_content(link, timeout=timeout)
                    if not content:
                        progress.update(article_task, advance=1)
                        continue
                    
                    # Generate multiple Q&A pairs with timeout
                    qa_pairs = generate_questions_and_answers(title, content, client)
                    if qa_pairs:
                        for qa_pair in qa_pairs:
                            articles.append(Article(
                                title=title,
                                link=link,
                                date=date,
                                content=content,
                                question=qa_pair['question'],
                                answer=qa_pair['answer'],
                                answer_context=qa_pair['answer_context']
                            ))
                except TimeoutError:
                    console.print(f"[yellow]Timeout processing article: {title}[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error processing article {title}: {str(e)}[/red]")
                
                progress.update(article_task, advance=1)
            
            progress.remove_task(article_task)
            progress.update(feed_task, advance=1)
    
    # Evaluate and filter articles
    indices_to_keep = evaluate_questions(articles, client)
    return [articles[i] for i in indices_to_keep]

def save_dataset(articles: List[Article], filename: str = "news_questions.json"):
    """Save the generated questions and answers to a JSON file."""
    dataset = [
        {
            "id": idx,
            "question": article.question,
            "answer": article.answer,
            "answer_context": article.answer_context,
            "title": article.title,
            "link": article.link,
            "date": article.date,
            "content": article.content
        }
        for idx, article in enumerate(articles)
    ]
    
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    console.print(f"[green]Dataset saved to {filename} with {len(articles)} entries[/green]") 