import re 
import emoji
import numpy as np 
import pandas as pd 
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('transformer_mdl')


def extract_txt(file_reader):
    txt = ''
    for page in reader.pages:
        page_txt = page.extract_text()
        if page_txt: 
            txt += page_txt + ' '
            return txt

def filter_txt(txt):
    url_pattern = r'(https?://\S+|www\.\S+)|((http|https)\s*:\s*//\s*\S+)'
    txt = txt.lower()
    txt = emoji.replace_emoji(txt,'')
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub(url_pattern,'',txt)
    txt = txt.replace('\n',' ')
    txt = txt.strip()
    return txt

def evaluate_score(job_txt,resume_txt):
    job_desc_emb = model.encode(job_txt,convert_to_numpy=True,normalize_embeddings=True)
    resume_emb = model.encode(resume_txt,convert_to_numpy=True,normalize_embeddings=True)
    similarity_score = cosine_similarity([job_desc_emb],[resume_emb])[0]
    return similarity_score

# Streamlit App UI
st.set_page_config(page_title='Resume Screening')
st.title('Resume Screening')
st.write('Enter the job description 👇')
user_inp = st.text_area('Job description here : ')
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    with open(f'uploads/{uploaded_file.name}', 'wb') as file:
        file.write(uploaded_file.getbuffer())

    reader = PdfReader(f'uploads/{uploaded_file.name}')
    extrected_txt = extract_txt(reader)
    resume_txt = filter_txt(extrected_txt)
    job_desc = filter_txt(user_inp)
    score = evaluate_score(job_desc,resume_txt)
    
    if score >= 0.70:
        st.success('Strong match. This resume aligns well with the job requirements. Proceed with shortlisting.')
    elif 0.55 <= score < 0.70:
        st.warning('Partial match. This candidate fits some requirements but needs manual review before shortlisting.')
    else:
        st.error('This resume does not match the job requirements. Do not shortlist.”')




            