from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import streamlit as st

@st.cache(show_spinner=False)
def topic_load_model(model_dir="facebook/bart-large-mnli"):
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return pipeline("zero-shot-classification", tokenizer=tokenizer, model=model)


@st.cache(show_spinner=False)
def sentiment_load_model(model_dir="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True):
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return pipeline("sentiment-analysis", tokenizer=tokenizer, model=model, return_all_scores=return_all_scores)


def topic_predict(model,
              doc,
              labels,
              multilabel=True,
              max_length=1024,
              batch_size=8,
              include_labels=True):

    output = model(doc, labels, "This text is about {}.", multilabel)
    df = pd.DataFrame({0:output["scores"]}).T
    df.columns = output['labels']
    return df


def sentiment_predict(model, doc):
    pred = model(doc)
    pred = pd.DataFrame({list(dic.values())[0]: list(dic.values())[1] for dic in pred[0]}, index=[0])
    return pred


def predict(topic_model, sentiment_model, doc, topics, multi_topics=True, show_progress=True):
    
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
        topic_pred.append(topic_predict(model=topic_model, doc=d, labels=topics, multilabel=multi_topics))
        sent_pred.append(sentiment_predict(model=sentiment_model, doc=d))
        if show_progress:
            progress_bar.progress((i+1)/max_value)
            print(i)
    
    # return in dictonary format
    return {"Topic":pd.concat(topic_pred).reset_index(drop=True),
            "Sentiment":pd.concat(sent_pred).reset_index(drop=True)
            }

@st.cache
def format_topic_sent(dic, doc):
    """
    add original text, topic predictions and sentiment predictions for each paragraph
    """
    df = pd.concat(list(dic.values()), axis=1)
    df.insert(0, "Original text", doc, True)
    df.reset_index(inplace=True, drop=True)
    return round(df * 100,3)

@st.cache
def compute_weighted_sentiment(dic):

    """
    Compute weighted sentiment predictions
    """

    topic_pred = dic["Topic"]
    sent_pred = dic["Sentiment"]
    sent_weighted = [topic_pred.multiply(sent_pred[c], axis="index")  for c in sent_pred.columns]
    sent_weighted_conc = pd.concat(sent_weighted, axis=1, keys=sent_pred.columns)
    return (pd.DataFrame(sent_weighted_conc.sum().div(topic_pred.sum(), level=1), columns=["Confidence (%)"])*100).round(3)