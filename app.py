import streamlit as st
from models.transformers import run_transformers
from models.lda import run_lda
def main():
    
    st.sidebar.title("Menu")
    option = st.sidebar.selectbox(
        'Choose application',
         ('Transformers', 'LDA'))

    if option == "Transformers":
        run_transformers()
    
    elif option == "LDA":
        run_lda()
        
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()