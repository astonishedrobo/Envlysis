import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from analyze_news import analyze_news
import os
from dotenv import load_dotenv
import shutil
import threading
import time
from argparse import ArgumentParser

# Global lock for file writing
file_lock = threading.Lock()

def process_news_item(item, index, questions, model_name ='gpt-3.5-turbo', max_retries=2):
    for attempt in range(max_retries):
        try:
            temp = {
                'title': item['title'],
                'news_idx': index,
                'analysis': analyze_news(item['article'], questions, model_name=model_name)
            }
            
            # Save individual result to JSON file
            with file_lock:
                with open(f'news_individual/news_analysis_{index}.json', 'w') as f:
                    json.dump(temp, f, indent=4)
            
            return temp
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                print(f"Rate limit hit for item {index}. Retrying after 5 seconds...")
                time.sleep(5)
            else:
                print(f"Error processing item {index}: {e}")
                # Log the failed index
                with file_lock:
                    try:
                        with open('failed_news_indices.json', 'r+') as f:
                            failed_indices = json.load(f)
                            failed_indices['idx'].append(index)
                            f.seek(0)
                            json.dump(failed_indices, f, indent=4)
                    except FileNotFoundError:
                        with open('failed_news_indices_rerun.json', 'w') as f:
                            json.dump({'idx': [index]}, f, indent=4)
                return {
                    'title': item['title'],
                    'news_idx': index,
                    'analysis': None
                }
    
    # If we've exhausted all retries
    return {
        'title': item['title'],
        'news_idx': index,
        'analysis': None
    }

def process_batch(batch, questions, pbar, model_name='gpt-3.5-turbo'):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_news_item, item, idx, questions, model_name=model_name) 
                   for idx, item in batch]
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)
        
    return results

def main():
    args = ArgumentParser()
    args.add_argument("--model", type=str, default='gpt-3.5-turbo', help="Model name to use for analysis")
    args = args.parse_args()

    # Load the environment variables
    env_path = '.env'
    load_dotenv(dotenv_path=env_path)

    # Load the news
    with open('/home/soumyajit/Downloads/RA/Codes/data/agriculture_news.json', 'r') as f:
        news = json.load(f)

    # Load the questions
    with open('news_questions.json', 'r') as f:
        questions = json.load(f)

    # Create a directory to save individual results
    if os.path.exists('news_individual'):
        is_delete = input("The 'news_individual' directory already exists. Do you want to delete it? [y/N]: ")
        is_delete = 'N' if not is_delete else is_delete.upper()
        if is_delete == 'Y':
            shutil.rmtree('news_individual')
        else:
            print("Terminating. Folder 'news_individual' already exists.")
    os.makedirs('news_individual')

    
    batch_size = 50
    analysis = []
    with tqdm(total=len(news), desc="Processing news items") as pbar:
        for i in range(0, len(news), batch_size):
            batch = list(enumerate(news[i:i+batch_size], start=i))
            batch_results = process_batch(batch, questions, pbar, model_name=args.model)
            analysis.extend(batch_results)
        
            # Optional: Add a delay between batches to further mitigate rate limiting
            time.sleep(30)

    # Save the complete analysis in a JSON file
    with open('news_analysis_rerun.json', 'w') as f:
        json.dump(analysis, f, indent=4)

if __name__ == "__main__":
    main()