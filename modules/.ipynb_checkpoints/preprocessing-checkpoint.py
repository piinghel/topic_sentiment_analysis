import PyPDF2
import os
import requests 
import re
import string
import io
import numpy as np
import streamlit as st
import spacy
import gensim


#@st.cache(allow_output_mutation=True, show_spinner=False)
def extract_content(url=None, open_pdf_file=None):    
    """
    A simple user define function that, given a url or pdf file, downloads PDF text content
    Parse PDF and return plain text version
    """

    if url is not None and open_pdf_file is None:
        try:
            # retrieve PDF binary stream
            response = requests.get(url)
            open_pdf_file = io.BytesIO(response.content)
            # return concatenated content
        except:
            return np.nan
    try:
        pdf = PyPDF2.PdfFileReader(open_pdf_file)  
        # access pdf content
        text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
    except:
        return np.nan
    # return concatenated content
    return "\n".join(text)


def remove_non_ascii(text):
    """
    remove non ascii text
    """
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))


def not_header(line):
    """
    remove headers
    """
    # as we're consolidating broken lines into paragraphs, we want to make sure not to include headers
    return not line.isupper()


#@st.cache(allow_output_mutation=True, show_spinner=False)
def load_spacy_model(model="en_core_web_sm"):
    """
    load spacy model
    """
    
    spacy.cli.download(model)
    nlp = spacy.load(model, disable=['ner'])
    return nlp


def extract_statements(
    text=None, 
    nlp=None, 
    make_sentence=False, 
    n_min_word_paragraph=50, 
    n_max_word_paragraph=200
    ):
    """
    Extracting ESG statements from raw text by removing junk, URLs, etc.
    We group consecutive lines into paragraphs and use spacy to parse sentences.
    """
  
    # remove non ASCII characters
    text = remove_non_ascii(text)
    
    
    lines = []
    prev = ""
    n_words = 0
    for line in text.split('\n'):
        # aggregate consecutive lines where text may be broken down
        # only if next line starts with a space or previous does not end with punctation mark and between
        if((line.startswith(' ') or not prev.endswith(('.','?', '!'))) and n_words <= n_max_word_paragraph):
            prev = prev + ' ' + line
            n_words = len(prev.split())
        
        # min words in paragraph
        elif n_words <=n_min_word_paragraph:
            prev = prev + ' ' + line
            n_words = len(prev.split())
            
        else:
            # new paragraph
            lines.append(prev)
            prev = line
            n_words = 0
            
    # don't forget left-over paragraph
    lines.append(prev)
    # clean paragraphs from extra space, unwanted characters, urls, etc.
    # best effort clean up, consider a more versatile cleaner
    sentences = []
    for line in lines:
        
        # removing header number
        line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
        # removing trailing spaces
        line = line.strip()
        # words may be split between lines, ensure we link them back together
        line = re.sub('\\s?-\\s?', '-', line)
        # remove space prior to punctuation
        line = re.sub(r'\s?([,:;\.])', r'\1', line)
        # ESG contains a lot of figures that are not relevant to grammatical structure
        line = re.sub(r'\d{5,}', r' ', line)
        # remove mentions of URLs
        line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
        # remove multiple spaces
        line = re.sub('\\s+', ' ', line)
            
        # split paragraphs into well defined sentences using spacy
        if make_sentence:
            try:
                for part in list(nlp(line).sents):
                    part_strip =  str(part).strip()
                    # remove senteces with only 30 characters
                    if len(part_strip) > 30:
                        sentences.append(part_strip)
            except ValueError:
                 print("Check if nlp model was loaded")
        else:
            sentences.append(line)
    
    return sentences


def tokenize(sentence):
    gen = gensim.utils.simple_preprocess(sentence, deacc=True)
    return ' '.join(gen)


def lemmatize(text, nlp):
  
    # parse sentence using spacy
    doc = nlp(text) 

    # convert words into their simplest form (singular, present form, etc.)
    lemma = []
    for token in doc:
        if (token.lemma_ not in ['-PRON-']):
            lemma.append(token.lemma_)

    return tokenize(' '.join(lemma))


   