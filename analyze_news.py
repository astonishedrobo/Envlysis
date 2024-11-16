from utils.analysis.analyzer import analyze_doc
import json

def analyze_news(news: str, questions: json, model_name: str = 'gpt-3.5-turbo'):
    prompt = questions["prompt"]
    questions = questions["questions"]

    question = f"{prompt}\n\n" + "Questions: " + "\n\n".join(questions)
    # print(question)
    return analyze_doc(news, question, model_name=model_name)
    
    
    
        