import os
from PyPDF2 import PdfReader
import docx
import re
import pytesseract
from PIL import Image
import io
import fitz
from docx.document import Document
from docx.oxml.shape import CT_Picture
from docx.parts.image import ImagePart
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\u00a0', ' ')
    text = text.replace('\n', ' ').replace('\r', '')
    return text

def perform_ocr(image):
    return pytesseract.image_to_string(image)

def extract_pdf_text(file_path):
    text = ''
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
            
            for img in page.get_images():
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                text += ' ' + perform_ocr(image)
    
    return clean_text(text)

def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    for rel in doc.part.rels.values():
        if isinstance(rel._target, ImagePart):
            image_bytes = rel._target.blob
            image = Image.open(io.BytesIO(image_bytes))
            text += ' ' + perform_ocr(image)
    
    return clean_text(text)

def extract_txt_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return clean_text(file.read())

def extract_image_text(file_path):
    image = Image.open(file_path)
    text = perform_ocr(image)
    return clean_text(text)

def upload_and_parse_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return extract_pdf_text(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_docx_text(file_path)
    elif file_extension == '.txt':
        return extract_txt_text(file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return extract_image_text(file_path)
    else:
        raise ValueError('Unsupported file type')
    
nltk.download('punkt_tab')

def tokenize_text(text):
    return word_tokenize(text.lower())

def vectorize_text(text, word2vec_model):
    tokens = tokenize_text(text)
    word_vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_vectors, axis=0)


def process_document_vector(file_path):
    text = upload_and_parse_document(file_path)
    
    tokenized_text = [tokenize_text(text)]
    
    word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)
    
    text_vector = vectorize_text(text, word2vec_model)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_word_vectors = []
    
    for word in tfidf_feature_names:
        if word in word2vec_model.wv:
            word_vector = word2vec_model.wv[word]
            tfidf_score = tfidf_matrix[0, tfidf_vectorizer.vocabulary_[word]]
            tfidf_word_vectors.append(word_vector * tfidf_score)
    
    if tfidf_word_vectors:
        final_vector = np.mean(tfidf_word_vectors, axis=0)
    else:
        final_vector = np.zeros(word2vec_model.vector_size)
    
    return final_vector

def get_vector_db(file_path):
    text = upload_and_parse_document("Pre-Implementation Design Report.pdf")
    embeddings = GPT4AllEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    persist_directory = "./chroma_db"
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectordb

def document_pipeline(file_path):
    text = upload_and_parse_document(file_path)
    final_vec = process_document_vector(file_path)
    vector_db = get_vector_db(file_path)
    return text, final_vec, vector_db