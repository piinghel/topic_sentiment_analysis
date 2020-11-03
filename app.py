import streamlit as st
from models.transformers import run_transformers

def main():
    
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox(
        'Choose application',
         ('Transformers', 'LDA'))

    if option == "Transformers":
        run_transformers()
    
    elif option == "LDA":
        st.header("TO BE DONE")
        
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()