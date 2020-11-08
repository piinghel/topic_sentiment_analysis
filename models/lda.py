import streamlit as st
import modules.preprocessing as preprocess
import joblib
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px


@st.cache(show_spinner=False)
def preprocess_wrapper(url, open_pdf_file, make_sentence=False):
    """
    preprocess text
    """
    data = preprocess.extract_content(url=url, open_pdf_file=open_pdf_file)
    data = preprocess.extract_statements(text=data, make_sentence=make_sentence)
    return data


@st.cache(show_spinner=False)
def predict(data, word_vec, lda):
    """
    make predictions
    """
    word_tf_new = word_vec.transform(data)
    transformed = lda.transform(word_tf_new)
    return transformed


def plot_topic_dist(predictions, df):
    """
    plots topic distribution
    """
    st.subheader("Topic distribution of new document")
    df_prob_topic = pd.DataFrame(predictions, columns=list(range(1, 10 + 1)))
    out = df_prob_topic.multiply(df["nr_words"], axis="index").sum() / df["nr_words"].sum()
    df_w = pd.DataFrame(out * 100, columns=["probability"]).sort_values('probability', ascending=False)
    df_w["topic"] = df_w.index
    df_w.reset_index(inplace=True, drop=True)
    fig = px.bar(df_w, x='topic', y='probability', text='probability')
    fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)


def run_lda():
    
    st.header("Latent Dirichlet Allocation (LDA)")
    example = "https://impact.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/jpmc-cr-esg-report-2019.pdf"
    url_report = st.text_input("Enter url of esg report", example)
    # make html
    st.subheader("Topic distribution from trained LDA model")
    HtmlFile = open("output/output_filename.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height = 800)

    # load word vectorizer
    filename_count = 'trained_models/CountVectorizer.sav'
    word_tf_vectorizer = joblib.load(filename_count)

    # load lda model
    filename_lda = 'trained_models/lda_fitted.sav'
    lda_model = joblib.load(filename_lda)
    nlp = preprocess.load_spacy_model()

    # preprocess
    data = preprocess_wrapper(url=url_report, open_pdf_file=None, make_sentence=False)
    df1 = pd.DataFrame(data, columns=["text"])
    df1["lem"] = df1['text'].apply(preprocess.lemmatize, nlp=nlp)
    
    # predict
    transformed = predict(data=df1["lem"], word_vec=word_tf_vectorizer, lda=lda_model)

    # topic with highest probability
    # add 1 (start otherwise at 0)
    a = [np.argmax(distribution) + 1 for distribution in transformed]
    # with associated probability
    b = [np.max(distribution) for distribution in transformed]

    # consolidate LDA output into a handy dataframe 
    df1["nr_words"] = df1["lem"].apply(lambda x: len(x.split()))
    df2 = pd.DataFrame(zip(a, b , transformed), columns=['topic', 'probability', 'probabilities'])
    esg_group = pd.concat([df1, df2], axis=1)

    # topic distribution
    plot_topic_dist(predictions=transformed, df=df1)

    # display dataframe
    with st.beta_expander("Expand results"):
        st.table(esg_group[["lem","topic","probability"]])

    # topic probabilities
    with st.beta_expander("Expand topic distribution"):
        fig = px.histogram(esg_group, x="probability")
        st.plotly_chart(fig, use_container_width=True) 

    # Detail results for a topic
    with st.beta_expander("Expand statement per topic"):
        topic_nr = st.number_input("Give topic number", min_value=1, max_value=10, value=6)
        esg_subset = esg_group[esg_group.topic==topic_nr].sort_values(by='probability', ascending=False)
        st.table(esg_subset[["lem","topic","probability"]])

    # Highest probability statement for each topic
    with st.beta_expander("Expand highest probability statement for each topic"):
        a = esg_group.loc[esg_group.groupby('topic')['probability'].idxmax()]
        st.table(a[["lem","topic","probability"]])