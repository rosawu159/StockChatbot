import streamlit as st
import json
import requests
import re
from pprint import pp
import pandas as pd
import numpy as np
import tiktoken
from PyPDF2 import PdfReader


def get_completion(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=1000):
  payload = { "model": model, "temperature": temperature, "messages": messages, "max_tokens": max_tokens }
  headers = { "Authorization": f'Bearer {API_KEY}', "Content-Type": "application/json" }
  response = requests.post('https://api.openai.com/v1/chat/completions', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  if response.status_code == 200 :
    return obj["choices"][0]["message"]["content"]
  else :
    return obj["error"]

def get_embeddings(input, model="text-embedding-ada-002"):
  payload = { "input": input, "model": model }
  headers = { "Authorization": f'Bearer {API_KEY}', "Content-Type": "application/json" }
  response = requests.post('https://api.openai.com/v1/embeddings', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  if response.status_code == 200 :
    return obj["data"][0]["embedding"]
  else :
    return obj["error"]
  
def cosine_similarity(a, b):
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(cosine)
    return cosine

def search_similar(df, query, n=3):
    query_embedding = get_embeddings(query)
    df["similarity"] = df['embeddings'].apply(lambda x: cosine_similarity(x, query_embedding))
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )
    print(results)
    return results


# Load the model of choice
def load_data(file_name):
    pdf_reader = PdfReader(file_name)
    # Text variable will store the pdf text
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_into_many(text, max_tokens = 500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # 使用 re.split 來拆分字符串，並用 filter 函數去除空字符串
    pattern = r'[。.]' # 中文。和英文.
    sentences = list(filter(None, re.split(pattern, text)))

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []
    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

# Set the title for the Streamlit app
st.title("Stock chatbot")
API_KEY = st.text_input("Enter your API key", type="password")
uploaded_file = st.file_uploader("Upload File", type="pdf")
user_question = st.text_input("Enter your question:")
submit_button = st.button('Submit')

if submit_button and user_question:
    print(uploaded_file.name)
    content = load_data(uploaded_file)
    chunks = split_into_many(content)
    embeddings = []
    for n, chunk in enumerate(chunks):
        print("for chunk in chunks", n, len(chunk), len(chunks))
        emb = get_embeddings(chunk)
        embeddings.append(emb)
    df = pd.DataFrame({
        'chunk': chunks,
        'embeddings': embeddings
    })
    docs = search_similar(df, user_question)
    context = ''
    for chunk in docs["chunk"]:
        context += chunk + "\n"
    prompt= f'''
        Answer the question based on the context below,
        and if the question can't be answered based on the context, say "I don't know"

        Context: {context}

        ---

        Question: {user_question}
        Answer:'''
    result = get_completion([ {"role": "user", "content": prompt }], model="gpt-3.5-turbo")
    st.write("Answer:", result)



