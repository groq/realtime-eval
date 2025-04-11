from .core.feed_handler import load_feeds, fetch_feed, display_articles

def main():
    for feed_info in load_feeds():
        feed = fetch_feed(feed_info['url'])
        display_articles(feed, feed_info['name'])

if __name__ == "__main__":
    main() 