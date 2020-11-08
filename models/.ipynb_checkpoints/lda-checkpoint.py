import streamlit as st
import modules.preprocessing as preprocess
import joblib
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
import streamlit.components.v1 as components


@st.cache(show_spinner=False)
def preprocess_wrapper(url, open_pdf_file, make_sentence=False):
    data = preprocess.extract_content(url=url, open_pdf_file=open_pdf_file)
    data = preprocess.extract_statements(text=data, make_sentence=make_sentence)
    return data

def run_lda():
    
    example = "https://impact.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/jpmc-cr-esg-report-2019.pdf"
    url_report = st.text_input("Enter url of esg report", example)
    nlp = preprocess.load_spacy_model()
    data = preprocess_wrapper(url=url_report, open_pdf_file=None, make_sentence=False)
    
    df_text = pd.DataFrame(data, columns=["text"])
    df_lem = df_text["text"].apply(preprocess.lemmatize, nlp=nlp)
    
    # load word vectorizer
    filename_count = 'trained_models/CountVectorizer.sav'
    word_tf_vectorizer = joblib.load(filename_count)

    # load lda model
    filename_lda = 'trained_models/lda_fitted.sav'
    lda_model = joblib.load(filename_lda)
    
    # predict
    word_tf_vectorizer_new = word_tf_vectorizer.transform(df_lem)
    
    # visulaize
    vis_data = pyLDAvis.sklearn.prepare(lda_model, word_tf_vectorizer_new, word_tf_vectorizer, mds='tsne')
    pyLDAvis.save_html(vis_data, 'output/output_filename.html')
    
    # make html
    HtmlFile = open("output/output_filename.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height = 800)
    
    
    
    