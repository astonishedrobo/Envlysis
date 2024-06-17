import argparse
import dotenv
import os
from utils.analysis.analyzer import analyze_doc

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the documentation of a project.')
    parser.add_argument('--doc_path', type=str, help='The path to the project to analyze.')
    parser.add_argument('--prompt_path', type=str, help='The path to the prompt file (txt).')
    parser.add_argument('--augment_links', type=bool)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the .env file
    dotenv_path = os.path.join(os.getcwd(), '.env')
    dotenv.load_dotenv(dotenv_path)

    # Load the prompt
    with open(args.prompt_path, 'r') as file:
        prompt = file.read()

    # Analyze the documentation
    doc_analysis = analyze_doc(args.doc_path, question=prompt, augment_link=args.augment_links)
    print(doc_analysis)

if __name__ == "__main__":
    main()
