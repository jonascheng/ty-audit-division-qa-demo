import os
import logging
import streamlit as st
from streamlit_extras.mention import mention
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Import proprietory module
import config.env
import config.logging
from dto import const
from query import web, embeddings, qa
from util.openai import chatter, memory

# Get logger
logger = logging.getLogger(__name__)


# Return indexer base on target name
def get_indexer(target_name: str):
    from query.embeddings import QueryEmbeddings

    if target_name == 'investigation':
        collection_name = os.environ.get(
            'EMBEDDINGS_INVESTIGATION_REPORTS_COLLECTION_NAME')
    else:
        collection_name = os.environ.get(
            'EMBEDDINGS_TAIWAN_LAW_COLLECTION_NAME')

    return QueryEmbeddings(
        vectorstore_filepath=os.environ.get('EMBEDDINGS_FILEPATH'),
        collection_name=collection_name)


# Function for querying Taiwan law database
def search_vector_store(
        prompt_input,
        indexer,
        chain_type: str = 'stuff',) -> str:
    from query import qa
    from util.openai import chatter

    # create retrieval qa
    rqa = qa.EmbeddingsRetrievalQA(
        llm=chatter(),
        chain_type=chain_type,
        retriever=indexer.as_multiquery_retriever(),
        return_source_documents=True)

    search_results = rqa.query({"query": prompt_input})

    return {
        "answer": search_results['result'],
        "source_documents": search_results['source_documents']
    }


def handle_selectbox_change():
    if st.session_state.target_name == const.APP_QUERY_TARGET_LAW:
        if "law_indexer" not in st.session_state.keys():
            st.session_state.law_indexer = get_indexer('law')
    elif st.session_state.target_name == const.APP_QUERY_TARGET_INVESTIGATION:
        if "investigation_indexer" not in st.session_state.keys():
            st.session_state.investigation_indexer = get_indexer('investigation')


st.set_page_config(page_title=const.APP_TITLE, page_icon='💬')

with st.sidebar:
    st.title(const.APP_TITLE)

    # a drop down for selecting the query target vector store
    target_name = st.selectbox(
        "選擇查詢目標",
        const.APP_QUERY_TARGETS,
        index=None,
        key='target_name',
        on_change=handle_selectbox_change)

    # an input for the user to enter a query
    query_input = st.text_input("輸入查詢", key='query_input')

    # a button to submit the query
    submit_button = st.button("送出查詢", key='submit_button')

# Check if button is clicked
if submit_button:
    if not target_name or not query_input:
        st.error("請選擇查詢目標並輸入查詢")
    else:
        with st.spinner("檢索中..."):
            if st.session_state.target_name == const.APP_QUERY_TARGET_LAW:
                result_set = search_vector_store(
                    prompt_input=query_input,
                    indexer=st.session_state.law_indexer,
                    chain_type='stuff')
            else:
                result_set = search_vector_store(
                    prompt_input=query_input,
                    indexer=st.session_state.investigation_indexer,
                    chain_type='map_reduce')

        st.write(result_set)