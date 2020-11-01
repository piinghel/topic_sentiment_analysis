from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import streamlit as st
import numpy as np

#@st.cache(show_spinner=False)
def topic_load_pipeline(directory="facebook/bart-large-mnli"):
    # load models
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory)
    return pipeline("zero-shot-classification", tokenizer=tokenizer, model=model)


#@st.cache(show_spinner=False)
def sentiment_load_pipeline(directory="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True):
    # load models
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory)
    return pipeline("sentiment-analysis", tokenizer=tokenizer, model=model, return_all_scores=return_all_scores)

#@st.cache(show_spinner=False)
def load_pipeline(dir_topic, dir_sent):
    topic_pipeline = topic_load_pipeline(directory=dir_topic)
    sentiment_pipeline = sentiment_load_pipeline(directory=dir_sent)
    return topic_pipeline, sentiment_pipeline


def topic_predict(pipeline,
                  doc,
                  labels,
                  multilabel=True,
                  batch_size=32,
                  include_labels=True):

    output = pipeline(doc, labels, "This text is about {}.", multilabel)
    df = pd.DataFrame({0:output["scores"]}).T
    df.columns = output['labels']
    return df


def sentiment_predict(pipeline, doc):
    pred = pipeline(doc)
    pred = pd.DataFrame({list(dic.values())[0]: list(dic.values())[1] for dic in pred[0]}, index=[0])
    return pred


def predict(topic_pipeline, sentiment_pipeline, doc, topics, multi_topics=True, show_progress=True):
    
    """
    Wrapper for topic and sentiment predictions
    """
    topic_pred = [] 
    sent_pred = []
    

    # loop over document
    if show_progress:
        max_value = len(doc)
        progress_bar = st.progress(0)
    for i, d in enumerate(doc):
        topic_pred.append(topic_predict(pipeline=topic_pipeline, doc=d, labels=topics, multilabel=multi_topics))
        sent_pred.append(sentiment_predict(pipeline=sentiment_pipeline, doc=d))
        if show_progress:
            progress_bar.progress((i+1)/max_value)
            #print(i)
    
    # return in dictonary format    
    return {"Topic":pd.concat(topic_pred).reset_index(drop=True),
            "Sentiment":pd.concat(sent_pred).reset_index(drop=True)
            }


def format_topic_sent(dic, doc):
    """
    add original text, topic predictions and sentiment predictions for each paragraph
    """
    df = pd.concat(dic, axis=1)
    df.insert(0, "Original text", np.array(doc), True)
    df.reset_index(inplace=True, drop=True)
    return df

def compute_weighted_sentiment(dic, nr_words, weighted_by_words=True):

    """
    Compute weighted sentiment predictions by words (optional)
    """
    
    # extract topic and sentiment dataframe
    topic_pred = dic["Topic"]
    sent_pred = dic["Sentiment"]
    
    if weighted_by_words:
        topic_weighted_words = topic_pred.multiply(nr_words, axis="index")
    
    sent_weighted = []
    for c in sent_pred.columns:
            # weight by words
            if weighted_by_words:
                sent_weighted.append(topic_pred.multiply(sent_pred[c], axis="index").multiply(nr_words, axis="index"))
            else:
                sent_weighted.append(topic_pred.multiply(sent_pred[c], axis="index"))
    
    # concat to dataframe
    sent_weighted_conc = pd.concat(sent_weighted, axis=1, keys=sent_pred.columns)
    
    # normalize 
    if weighted_by_words:
        return (pd.DataFrame(sent_weighted_conc.sum().div(topic_weighted_words.sum(), level=1), columns=["Confidence (%)"])*100).round(3)
    else:
        return (pd.DataFrame(sent_weighted_conc.sum().div(topic_pred.sum(), level=1), columns=["Confidence (%)"])*100).round(3)