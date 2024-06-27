from PyPDF2 import PdfReader
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
import requests

    
def read_pdf(file_path: str, return_list: bool = False):
    pdf = PdfReader(file_path)
    text = []
    for page in pdf.pages:
        text = page.extract_text()
        text.append(text)

    if return_list:
        return text
    else:
        return '\n\n'.join(text)


def extract_hyperlinks(pdf_reader):
    links = set()
    for page in pdf_reader.pages:
        if '/Annots' in page:
            for annot in page['/Annots']:
                annot_obj = annot.get_object()

                if '/URI' in annot_obj.get('/A', {}):
                    uri = annot_obj['/A']['/URI']
                    links.add(uri)
    return links

def augment_link_content(file_path: str):
    """
    Aguments the contet with the content from the links.
    """
    pdf = PdfReader(file_path)

    # Find all the links
    links = extract_hyperlinks(pdf)

    # Extract Hyperlink Contents
    augmentation_text = "Extra Details: \n\n"
    for link in links:
        if link.endswith(".pdf"):
            response = requests.get(link)
            if response.status_code != 200:
                continue

            os.makedirs("file_cache", exist_ok=True)
            with open("file_cache/cache.pdf", "wb") as file:
                file.write(response.content)

            try:
                augmentation_text += read_pdf("file_cache/cache.pdf") + '\n'
            except:
                pass
            shutil.rmtree("file_cache")

    return augmentation_text

def split_and_store_db(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text])

    vector_db = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
    return vector_db

def join_context(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

def get_llm(model_name: str = 'gpt-3.5-turbo'):
    return ChatOpenAI(model_name=model_name)


def analyze_doc(file_path: str, question: str, augment_link: str = False, previous_context: str = None, model_name: str = 'gpt-3.5-turbo'):
    text = read_pdf(file_path)
    if augment_link:
        print("Augmenting Text With Content From Hyperlinks")
        text += augment_link_content(file_path)

    vector_db = split_and_store_db(text)
    
    # Retrieve the most similar chunks
    retriever = vector_db.as_retriever(search_type='similarity')

    # Define llm
    llm = get_llm(model_name=model_name)

    # Define prompt
    template = '''Use the following pieces of context to answer the question at the end in JSON format.
        If you don't know the answer, just retun None as value, don't try to make up an answer.
        

        Context: {context}

        Question: {question}

        Answer: 
    '''
    prompt = PromptTemplate.from_template(template)

    # Rag Chain
    rag_chain = (
        {"context": retriever | join_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )

    # Run the chain
    if previous_context:
        question = previous_context + '\n\n' + question
    answer = rag_chain.invoke(question)

    return answer


