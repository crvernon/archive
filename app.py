import io
import os

import numpy as np
import pandas as pd
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def load_db():
    """Load vector database"""

    embeddings = OpenAIEmbeddings()

    db_dir = "./data/archive"

    return Chroma(persist_directory=db_dir, embedding_function=embeddings)


def result_to_df(docs):
    """Convert a similarity query result to a data frame."""

    d = {"source": [], "page_number": [], "text_excerpt": []}
    for i in docs:
        extracted_text = i.page_content.replace("\n", "")

        text_excerpt = f"...{extracted_text}..."

        metadata = i.metadata
        
        d["source"].append(os.path.basename(metadata["source"]))
        d["page_number"].append(metadata["page"])
        d["text_excerpt"].append(text_excerpt)
        
    return pd.DataFrame(d)


def clear_text():
    """Clear text from entry"""

    st.session_state.user_input = ""


# load vectorstore
if "db" not in st.session_state:
    database = load_db()
    st.session_state.db = database

if "query" not in st.session_state:
    st.session_state.query = ""

if "n_results" not in st.session_state:
    st.session_state.n_results = 4

if "retreiver" not in st.session_state:
    st.session_state.retreiver = None

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

if "input" not in st.session_state:
    st.session_state["input"] = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)

# page title
st.title("Global Change Analysis Model (GCAM) Research Archive")

# description under title
st.markdown(
    """
    GCAM Archive

    See our [GCAM documentation](https://github.com/JGCRI/gcam-core) GitHub resource for more info!
    """
)

# search container
search = st.container()
search.markdown("#### Search for relevant documents:")

st.session_state.n_results = search.selectbox(
        'Select the number of results to generate:',
        (
            5,
            10, 
            15, 
            20,
            50,
            100,
        ),
        index=0
)

# set up retriever
st.session_state.retriever = st.session_state.db.as_retriever(
    search_type="similarity",
    # search_type="mmr",
    search_kwargs={
        "k": st.session_state.n_results, 
        # "search_distance": 0.1
    }
)

# query from user
st.session_state.query = search.text_input("Enter your query here:")

# get similary results from vector db
if len(st.session_state.query) > 0:

    docs = st.session_state.retriever.get_relevant_documents(st.session_state.query)

    # format query result
    st.session_state.query_result = result_to_df(docs)

if st.session_state.query_result is not None:
    search.write(st.session_state.query_result)

# export to CSV file
if st.session_state.query_result is not None:
    export = st.container()
    export.markdown("###### Export query results to CSV file:")
    bio = io.BytesIO()
    st.session_state.query_result.to_csv(bio, index=False)

    export.download_button(
            label="Export to CSV",
            data=bio.getvalue(),
            file_name="query_results.csv",
            mime="csv"
        )


qa = st.container()
qa.markdown("#### Conduct QA with the GCAM archive")

if st.session_state.qa_chain is None and st.session_state.retriever is not None:

    # st.session_state.qa_chain = RetrievalQA.from_llm(
    #     llm=ChatOpenAI(
    #         temperature=0.0,
    #         model_name="gpt-4",
    #         max_tokens=500
    #     ),
    #     retriever=st.session_state.retriever,
    #     return_source_documents=True
    # )

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0.0,
            model_name="gpt-4",
            max_tokens=500
        ),
        retriever=st.session_state.retriever,
        # memory=memory,
        return_source_documents=True
    )

# enter chat here
user_input = qa.text_input(
    "Your input to the chat", 
    st.session_state.input, 
    key="input", 
    placeholder="What would you like to discuss?", 
    on_change=clear_text,
    label_visibility='hidden'
)

if len(st.session_state.chat_history) > -1 and user_input:

    with st.spinner(f"Generating an answer to your question..."):
        
        output = st.session_state.qa_chain({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })

    st.session_state.chat_history.append(
        (user_input, output["answer"])
    )

    for i, o in reversed(st.session_state.chat_history):
        qa.info(i, icon="🧐")
        qa.success(o, icon="🤖")

    # --------------------------

    # with st.spinner(f"Generating an answer to your question..."):

    #     # generate QA response from query
    #     response = st.session_state.qa_chain(st.session_state.query)

    #     source_docs = []
    #     for source in response["source_documents"]:
    #         source_doc = os.path.basename(source.metadata["source"])
    #         source_page = source.metadata["page"]
    #         doc_string = f"{source_doc}, Page: {source_page}"

    #         if doc_string not in source_docs:
    #             source_docs.append(f"{source_doc}, Page: {source_page}")

    #     qa.markdown("###### Response")
    #     qa.markdown(f"""{response["result"]}""")

    #     qa.markdown("###### Sources")
    #     for i in source_docs:
    #         qa.markdown(f"""{i}""")

    # --------------------------






