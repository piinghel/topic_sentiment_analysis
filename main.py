import pandas as pd 
import PyPDF2
import os
import streamlit as st
import re
import string
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# own module
import helper_functions.models as models
import helper_functions.preprocessing as preprocess

# lay out
st.beta_set_page_config(layout="wide")

# title
st.title('Topic and sentiment analyzer')
uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

# specify models
model_dir_topic = st.text_input("Enter topic model directory", "joeddav/xlm-roberta-large-xnli")
model_dir_sent = st.text_input("Enter sentiment model directory", "distilbert-base-uncased-finetuned-sst-2-english")
st.markdown("**Click on the following [link](https://huggingface.co/models) to check out more models.**")   

# load models
with st.spinner("Loading models"):
    topic_model = models.topic_load_model(model_dir=model_dir_topic)
    sentiment_model = models.sentiment_load_model(model_dir=model_dir_sent)

# specify topics by the user
topics = st.text_input('Possible topics (separated by `,`)', "Environmental, Social, Governance, ESG", max_chars=1000)
topics = list(set([x.strip() for x in topics.strip().split(',') if len(x.strip()) > 0]))

# allow multiple topics to be correct (default is true)
multi_topics = st.checkbox('Allow multiple correct topics', value=True)

# preprocessing data
if uploaded_file is not None:
    with st.spinner("preprocessing data"):
        # return concatenated content
        text = preprocess.extract_content(open_pdf_file=uploaded_file)
        articles = preprocess.extract_statements(text=text)
    
    with st.beta_expander("Show processed paragraphs"):
        st.write(articles)

    n_paragraphs = len(articles)
    select_n = st.sidebar.slider("Select number of paragraphs", value=int(n_paragraphs/30+1),min_value=1, max_value=(n_paragraphs-1))
    
    if st.button('Make predictions'):
        predictions_all = models.predict(topic_model=topic_model, sentiment_model=sentiment_model, doc=articles[0:select_n], topics=topics, multi_topics=multi_topics)
        
        with st.beta_container():
            r1_c1, r1_c2, r1_c3 = st.beta_columns((1, 1.5, 1))
           
            # topic predictions
            r1_c1.subheader("Topic predictions")
            out_top = (pd.DataFrame(predictions_all["Topic"].mean(),columns=['Confidence (%)'])*100).sort_values(by='Confidence (%)',ascending=False)
            r1_c1.dataframe(out_top.style.format("{:.3}"))
            
            # Show weighted normalized sentiment predictions
            r1_c2.subheader("Weighted normalized sentiment predictions")
            out_sent = models.compute_weighted_sentiment(dic=predictions_all)
            r1_c2.dataframe(out_sent)
            
            # average sentiment
            r1_c3.subheader("Average sentiment")
            out_avg_sent = pd.DataFrame(predictions_all['Sentiment'].mean(), columns=["Confidence (%)"])*100
            r1_c3.dataframe(out_avg_sent)
            
        with st.beta_container():

            # all individual predictions
            r2_c1 = st.beta_columns((1))[0] 
            out_all = models.format_topic_sent(dic=predictions_all, doc=articles[0:select_n])
            r2_c1.subheader("All predictions")
            r2_c1.dataframe(out_all)

else:
    st.error("No pdf file selected")