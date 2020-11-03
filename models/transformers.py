import pandas as pd 
import PyPDF2
import os
import streamlit as st
import re
import string
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# own modules
import modules.pipeline as pipeline
import modules.preprocessing as preprocess


def load_global_vars():
    """
    load global variables
    """
    global TOPIC_DIR
    global SENTIMENT_DIR
    global TOPICS
    global MULTI_TOPICS
    TOPIC_DIR = "joeddav/xlm-roberta-large-xnli"
    SENTIMENT_DIR = "distilbert-base-uncased-finetuned-sst-2-english"
    TOPICS = "Environmental, Social, Governance, ESG"
    MULTI_TOPICS = True


def upload_file():
    """
    upload pdf file
    """
    st.title('Topic and sentiment analyzer')
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    return uploaded_file


def input_pipeline_dir(topic, sent):
    """
    specify pipeline directories
    """
    
    topic_dir = st.text_input("Enter topic model directory", topic)
    sent_dir = st.text_input("Enter sentiment model directory", sent)
    st.markdown("**Click on the following [link](https://huggingface.co/models) to check out more models.**") 
    return topic_dir, sent_dir


def define_topics(topics, multi_topic):
    """
    let you specify topics and choose to allow multiple correct topics
    """
    topics = st.text_input('Possible topics (separated by `,`)', topics, max_chars=1000)
    allow_multi_topics = st.checkbox('Allow multiple correct topics', value=multi_topic)
    topics = list(set([x.strip() for x in topics.strip().split(',') if len(x.strip()) > 0]))
    return topics, allow_multi_topics


def wrapper_preprocess_steps(uploaded_file, text_split):
    """
    wrapper around the preprocessing steps
    """
    
    # read in pdf file
    text = preprocess.extract_content(open_pdf_file=uploaded_file)

    # process pdf file and split in paragraphs
    if text_split == 'Make paragraphs':
        
        min_n_words = st.sidebar.slider(
        "Minimum number of words in a paragraph", 
        value=50,
        min_value=0, 
        max_value=1024
        )
        
        max_n_words = st.sidebar.slider(
        "Maximum number of words in a paragraph", 
        value=100,
        min_value=50, 
        max_value=1024
        )

        articles = preprocess.extract_statements(
            text=text, 
            make_sentence=False,
            n_min_word_paragraph=min_n_words,
            n_max_word_paragraph=max_n_words)

    # process pdf file and split in sentences
    elif text_split == 'Make sentences': 

        # load spacy model for making senteces
        nlp = preprocess.load_spacy_model()
        articles = preprocess.extract_statements(
            text=text, 
            make_sentence=True,
            nlp=nlp   
        )
         
    return articles


@st.cache(allow_output_mutation=False, show_spinner=False)
def wrapper_predictions(
    output_topic_dir, 
    output_sent_dir, 
    articles, 
    topics, 
    n_texts, 
    allow_multi_topics
    ):
    
    """
    wrapper for making predictions
    """
    
    # load pipeline
    topic_pipeline, sentiment_pipeline = pipeline.load_pipeline(
        dir_topic=output_topic_dir, 
        dir_sent=output_sent_dir
    )

    # make predictions
    predictions_all = pipeline.predict(
        topic_pipeline=topic_pipeline, 
        sentiment_pipeline=sentiment_pipeline, 
        doc=articles[0:n_texts], 
        topics=topics, 
        multi_topics=allow_multi_topics                                        
    )
    
    return predictions_all



def format_output(predictions_all, weighted_by_words, n_texts,articles):
    """
    format output
    """
    
    with st.beta_container():
        r1_c1, r1_c2, r1_c3 = st.beta_columns((1, 1.5, 1))
        
        out_all = pipeline.format_topic_sent(dic=predictions_all, doc=articles[0:n_texts])
        out_all["nr_words"] = out_all["Original text"].apply(lambda x: len(x.split()))

        # topic predictions
        r1_c1.subheader("Topic predictions")
        out_top = (pd.DataFrame(predictions_all["Topic"].mean(),
                                columns=['Confidence (%)'])*100).sort_values(by='Confidence (%)',ascending=False)
        r1_c1.dataframe(out_top.style.set_precision(2))

        # Show weighted normalized sentiment predictions
        r1_c2.subheader("Weighted normalized sentiment predictions")
        out_sent = pipeline.compute_weighted_sentiment(dic=predictions_all, weighted_by_words=weighted_by_words, nr_words=out_all["nr_words"])
        r1_c2.dataframe(out_sent)

        # average sentiment
        r1_c3.subheader("Average sentiment")
        
        # weighted
        if weighted_by_words:
            sent_words = predictions_all['Sentiment'].multiply(out_all["nr_words"], axis="index").sum()
            sum_words = out_all["nr_words"].sum()
            out_avg_sent = pd.DataFrame(sent_words / sum_words, columns=["Confidence %"])
            out_avg_sent = (out_avg_sent * 100).sort_values(by="Confidence %", ascending=False).round(3)
        # simple average
        else:
            out_avg_sent = pd.DataFrame(predictions_all['Sentiment'].mean(), columns=["Confidence (%)"])*100
        r1_c3.dataframe(out_avg_sent.style.set_precision(2))

    with st.beta_container():
        # all individual predictions
        r2_c1 = st.beta_columns((1))[0] 
        r2_c1.subheader("All predictions")
        r2_c1.dataframe(out_all.style.set_precision(2))
        
    
def run_transformers():
    """
    run transformer
    """
    
    # load global vars
    load_global_vars()
    
    # upload file
    uploaded_file = upload_file()
    if not isinstance(uploaded_file, st.uploaded_file_manager.UploadedFile):
        st.error("Select pdf file!")
        return
    
    # get topic and sent directroy
    output_topic_dir, output_sent_dir = input_pipeline_dir(topic=TOPIC_DIR, sent=SENTIMENT_DIR)
    if len(output_topic_dir) == 0 or len(output_sent_dir) == 0:
        st.error('Enter directory for the topic and sentiment model.')
        return
    
    # get topics
    topics, allow_multi_topics = define_topics(topics=TOPICS, multi_topic=MULTI_TOPICS)
    if len(topics) == 0:
        st.error('Enter at least one possible topic to see predictions.')
        return
    
    # preprocessing options
    st.sidebar.subheader("Preprocessing settings")
    
    # select number of paragraphs sentences
    n_text_selected = st.sidebar.slider(
        "Select the fraction of paragraphs/senteces to be processed", 
        value=0.05,
        min_value=0.01, 
        max_value=1.0
    )
        
    # choose how to split
    text_split=st.sidebar.selectbox(
        'How to split text',
        ('Make paragraphs','Make sentences'),
    )
    
    with st.spinner("Preprocessing data..."):
        articles = wrapper_preprocess_steps(uploaded_file=uploaded_file, text_split=text_split)

    with st.beta_expander("Show processed paragraphs"):
            st.write(articles)

    st.sidebar.subheader("Output calculations settings")
    weighted_by_words = st.sidebar.checkbox("Weight predictions by number of words", value=True)

    
    # subsetting 
    n_texts = int(len(articles)*n_text_selected)
    
    # make predictions
    with st.spinner("Predicting..."):
        predictions_all = wrapper_predictions(
            output_topic_dir=output_topic_dir, 
            output_sent_dir=output_sent_dir, 
            articles=articles, 
            n_texts=n_texts,
            topics = topics, 
            allow_multi_topics = allow_multi_topics
        )

        # format output
        format_output(
            predictions_all=predictions_all, 
            weighted_by_words=weighted_by_words,
            n_texts=n_texts,
            articles=articles
        )
