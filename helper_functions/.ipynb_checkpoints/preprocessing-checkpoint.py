
import PyPDF2
import os
import requests 
import re
import string
import io
import streamlit as st
import spacy


#@st.cache(show_spinner=False)
def extract_content(open_pdf_file=None, url=None):
    
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
            return ""
    pdf = PyPDF2.PdfFileReader(open_pdf_file)  
    # access pdf content
    text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
    # return concatenated content
    return "\n".join(text)

def remove_non_ascii(text):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))

def not_header(line):
    # as we're consolidating broken lines into paragraphs, we want to make sure not to include headers
    return not line.isupper()

def load_spacy_model(model="en_core_web_sm"):
    # load spacy model
    spacy.cli.download(model)
    nlp = spacy.load(model, disable=['ner'])
    return nlp

#@st.cache(show_spinner=False)
def extract_statements(text=None, nlp=None, make_sentence=False, n=200):
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
        # only if next line starts with a space or previous does not end with punctation mark.
        if((line.startswith(' ') or not prev.endswith(('.','?', '!'))) and n_words < 25):
            prev = prev + ' ' + line
            n_words = len(prev.split())
        else:
            # new paragraph
            lines.append(prev)
            prev = line
            n_words = 0
            
    # don't forget left-over paragraph
    lines.append(prev)
    return lines
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
            # check if nlp model is loaded 
            if nlp is not None:
                for part in list(nlp(line).sents):
                    part_strip =  str(part).strip()
                    if len(part_strip) > 30:
                        sentences.append(part_strip)
        else:
            sentences.append(line)
    
    return sentences



def get_split(text1, n1=50, n2=75):
    
    l_total = []
    l_parcial = []
    if len(text1.split()) // n1 > 0:
        n = len(text1.split()) // n1
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:n2]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*n1:w*n1 + n2]
            l_total.append(" ".join(l_parcial))
    return l_total    