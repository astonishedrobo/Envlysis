import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
from utils.crawling.crawler import date_range, fetch_news, save_news

# Global lock for thread-safe file operations
file_lock = Lock()

def fetch_and_save_news(start: tuple, end: tuple, query: str, filename: str, file_lock: threading.Lock, fetch_article: bool = False):
    news = fetch_news(start, end, query, fetch_article=fetch_article)
    save_news(news, file_lock, filename)
    return len(news)

def main():
    query = 'Indian Climate'
    start_date = datetime.date(2021, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    delta = 7
    filename = 'news_full.json'

    if os.path.exists(filename):
        os.remove(filename)

    total_iterations = (end_date - start_date).days // delta + 1

    # Concurrent fetching
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(fetch_and_save_news, start, end, query, filename, file_lock, fetch_article=True)
                  for start, end in date_range(start_date, end_date, delta)]

        # tqdm for concurrent progress bar
        with tqdm(total=total_iterations, desc="Fetching News") as pbar:
            for future in as_completed(futures):
                num_articles = future.result()
                pbar.update(1)
                pbar.set_postfix({"Articles Fetched": num_articles})

if __name__ == "__main__":
    main()