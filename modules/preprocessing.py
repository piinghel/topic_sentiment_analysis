import PyPDF2
import os
import requests 
import re
import string
import io
import numpy as np
import pandas as pd
import streamlit as st
import spacy
import gensim
from tqdm import tqdm
import requests
from pdfminer.high_level import extract_text
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file


#@st.cache(allow_output_mutation=True, show_spinner=False)
def extract_content_PyPDF2(url=None, open_pdf_file=None):    
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
    try:
        printable = set(string.printable)
        return ''.join(filter(lambda x: x in printable, text))
    except: 
        return ""


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
        # remove special characters
        line =re.sub('[^A-Za-z]+', ' ', line)
        # remove standalone characters
        line = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', line)
        # if length of line is smaller than 20, skip
        if len(line) < 20:
            continue 
            
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
    """
    tokenize sentence
    """
    gen = gensim.utils.simple_preprocess(sentence, deacc=True)
    return ' '.join(gen)


def lemmatize(text, nlp, stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'], method=1):
    """
    lemmatization and remove stopwords
    """

    # parse sentence using spacy
    text = text.lower()
    doc = nlp(text)

    # convert words into their simplest form (singular, present form, etc.)
    lemma = []
    for token in doc:
        
        if method == 1:
            if str(token) not in stop_words and (token.lemma_ not in ['-PRON-']) and len(str(token)) > 3:
                lemma.append(token.lemma_)
        
        elif method == 2:
            if str(token) not in stop_words and (token.pos_  in allowed_postags) and len(str(token)) > 3:
                lemma.append(token.lemma_)
        else:
            ValueError("Method can only be 1 or 2")


    return tokenize(' '.join(lemma))



def download_reports(df, directory, update=True):
    """
    downloads reports and save it in the given directory
    """
    
    if update:
        could_download = []
        could_not_download = []
        with tqdm(total=df.shape[0]) as pbar:
            for _, row in df.iterrows():
                url = row["url"]
                file_name = row['company'] + "-"+ str(int(row['year'])) + ".pdf"
                file_name = file_name.replace(" ", "")
                pdf_fname = directory + file_name

                # download file and save
                try:
                    resp = requests.get(url)
                    with open(pdf_fname, 'wb') as f:
                        f.write(resp.content)
                        row["filename"] = file_name
                except:
                    could_not_download.append(row['company'].values.tolist()[0])
                    print(f"Could not download {row['company']}")
                pbar.update(1)
                could_download.append(row)
    
    return pd.concat(could_download, axis=1).T, could_not_download
    

def extract_content_pdfMiner(file_name, directory):
    """
    alternative way to extract pdf content
    """

    try:
        text = extract_text(directory + file_name)
    
    except:
        return np.nan
    
    return text


def to_paragraph_per_row(df, columns_to_keep=[]):

    """
    paragraph from list to row in a dataframe
    """

    paragraph_l = []
    for i, p_l in enumerate(df["paragraph"]):
        for p in p_l:
            # columns to keep and a paragraph each time
            data_dict = {}
            if len(columns_to_keep) > 0:
                for c in columns_to_keep:
                    data_dict[c] = df[c][i]

            data_dict["paragraph"] = p
            # construct data frame
            df_p_to_row = pd.DataFrame.from_dict(data_dict, orient='index').T
            paragraph_l.append(df_p_to_row)
    
    return pd.concat(paragraph_l).reset_index(drop=True)
    


def load_processed_text(
    data,
    dir_read_pdf, 
    file_processed_text,
    columns_to_keep = [],
    n_min_word_paragraph=50, 
    n_max_word_paragraph=125,  
    update=False,
    method_extract_content = "pdfMiner"
    ):
    """
    loads preprocessed text if available, otherwise 
    downalod/reads in pdf files from urls and perform some preprocessing
    """

    # read file if it exits
    if os.path.isfile(file_processed_text) and not update:
        df = pd.read_csv(file_processed_text, sep='\t')
    
    # perfrom data manipulations
    else:
        # to monitor progresss
        tqdm.pandas()
        df = data.copy()
        # 1) extract pdf from url
        
        # 1.a PyPF2
        if method_extract_content == "PyPDF2":
            try:
                
                df["article"] = df["url"].progress_apply(extract_content_PyPDF2, open_pdf_file=None)
            except:
                ValueError("Could not download urls via PyPDF2")
        
        # 1.b pdfminer
        elif method_extract_content == "pdfMiner":
            try:
                df["article"] = df["filename"].progress_apply(extract_content_pdfMiner, directory=dir_read_pdf)
            except:
                ValueError("Could read in filesnames via pdfMiner")
        else:
            print("method not available: available methods include PyPDF2 or pdfMiner")

        df = df.dropna().reset_index(drop=True)
        
        # 2) split pdf file into smaller chunks and perfrom some basic cleaning
        df["paragraph"] = df["article"].progress_apply(
            func=extract_statements, 
            make_sentence=False, 
            n_min_word_paragraph=n_min_word_paragraph, 
            n_max_word_paragraph=n_max_word_paragraph
        )
        df = df[df['paragraph'].map(lambda d: len(d)) > 0].reset_index(drop=True)
        df = to_paragraph_per_row(df=df, columns_to_keep=columns_to_keep)
        df.to_csv(file_processed_text, sep='\t', index=False, header=True)

    return df


def load_lemmatize(data, dir_file, stop_words, nlp, method=1, update=False):
    """
    loads lemmatized text if exists, otherswise performs lemmatization and removes stopwords
    """

    # read file if it exits
    if os.path.isfile(dir_file) and not update:
        df = pd.read_csv(dir_file, sep='\t', header=None)
        data["paragraph_cleaned"] = df.values

    else:
        # progress bar
        tqdm.pandas()
        # perfrom some cleaning (stopwords, lemmatize)
        data["paragraph_cleaned"] = data['paragraph'].progress_apply(
                                                        lemmatize, 
                                                        nlp=nlp, 
                                                        stop_words=stop_words, 
                                                        method=method)
        # save raw and cleaned text
        data["paragraph_cleaned"].to_csv(dir_file, sep='\t', index=False, header=False)
    
    return data


def load_bert_embeddings(
            text_dir="output/CSR_processed_raw.txt", 
            model="distiluse-base-multilingual-cased", 
            dir_embeddings="output/CRS_bert_Embeddings.npy",
            update=False):
    
    """
    performs bert embeddings
    """

    # read file if exist
    if os.path.isfile(dir_embeddings) and not update:
        embeddings = np.load(dir_embeddings)
    
    # make predictions otherwise
    else:
        embeddings = bert_embeddings_from_file(text_dir, model)
        np.save(dir_embeddings, embeddings)
    
    return embeddings