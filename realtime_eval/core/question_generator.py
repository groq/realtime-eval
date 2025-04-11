import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress
from groq import Groq
from .feed_handler import fetch_feed, format_date, load_feeds

console = Console()

@dataclass
class Article:
    title: str
    link: str
    date: str
    question: Optional[str] = None
    answer: Optional[str] = None

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

def evaluate_questions(articles: List[Article], client: Groq) -> List[int]:
    """Evaluate questions and answers using an LLM to determine which to keep."""
    indices_to_keep = []
    offset = 0
    for i in range(0, len(articles), 5):
        batch = articles[i:i+5]
        batch_content = [
            {
                "question": article.question,
                "answer": article.answer
            }
            for article in batch
        ]
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that evaluates questions and answers in JSON format to test an LLM's ability to access real-time information from news headlines. "
                            "Use the following guidelines:\n\n"
                            "1. Analyze each question and answer pair to determine if it meets the criteria of being clear, specific, and based on a real-time event. "
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
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            print(response.choices[0].message.content)
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            new_indices_to_keep = result.get('indices', [])
            indices_to_keep.extend([i + offset for i in new_indices_to_keep])
            offset += 5

            console.print(f"[green]Running indices to keep: {indices_to_keep}[/green]")
        except Exception as e:
            console.print(f"[red]Error evaluating questions: {e}[/red]")
            return []
    return indices_to_keep

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
            "title": article.title,
            "link": article.link,
            "date": article.date
        }
        for idx, article in enumerate(articles)
    ]
    
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    console.print(f"[green]Dataset saved to {filename} with {len(articles)} entries[/green]") 