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


st.session_state.citations = {
    0: {
        "file_name": "1-s2.0-0140988383900142-main.pdf",
        "title": "A long-term global energy- economic model of carbon dioxide release from fossil fuel use",
        "reference": "Edmonds, J. & Reilly, J. A long-term global energy- economic model of carbon dioxide release from fossil fuel use. Energy Economics 5, 74–88 (1983).",
        "citation": "Edmonds & Reilly, 1983"
    },
    1: {
        "file_name": "1-s2.0-S2211467X1830004X-main.pdf",
        "title": "Peak energy consumption and CO2 emissions in China's industrial sector",
        "reference": "Zhou, S., Wang, Y., Yuan, Z. & Ou, X. Peak energy consumption and CO2 emissions in China’s industrial sector. ENERGY STRATEGY REVIEWS 20, 113–123 (2018).",
        "citation": "Zhou et al., 2018" 
    },
    2: {
        "file_name": "1-s2.0-S0301421515001275-main.pdf",
        "title": "China's transportationenergyconsumptionandCO2 emissions froma global perspective",
        "reference": "Yin, X. et al. China’s transportation energy consumption and CO2 emissions from a global perspective. ENERGY POLICY 82, 233–248 (2015).",
        "citation": "Yin, et al., 2015"
    },
    3: {
        "file_name": "1-s2.0-S0301421517304469-main.pdf",
        "title": "Improving building energy efficiency in India: State-level analysis of building energy efficiency policies",
        "reference": "Yu, S. et al. Improving building energy efficiency in India: State-level analysis of building energy efficiency policies. ENERGY POLICY 110, 331–341 (2017).",
        "citation": "Yu et al., 2017"
    },
    4: {
        "file_name": "908307.pdf",
        "title": "Projection of the Spatially Explicit Land Use/Cover Changes in China, 2010–2100",
        "reference": "Yuan, Y., Zhao, T., Wang, W., Chen, S. & Wu, F. Projection of the Spatially Explicit Land Use/Cover Changes in China, 2010-2100. ADVANCES IN METEOROLOGY 2013, (2013).",
        "citation": "Yuan et al., 2013"
    },
    5: {
        "file_name": "Global_Change_Assessment_Model_GCAM_considerations_of_the_primary_sources_energy_mix_for_an_energetic_scenario_that_could_meet_Paris_agreement.pdf",
        "title": "Global Change Assessment Model (GCAM) considerations of the primary sources energy mix for an energetic scenario that could meet Paris agreement",
        "reference": "Lazarou, S., Christodoulou, C., Vita, V., & IEEE. Global Change Assessment Model (GCAM) considerations of the primary sources energy mix for an energetic scenario that could meet Paris agreement. in ASPETE - School of Pedagogical & Technological Education (2019).",
        "citation": "Lazarou, Christodoulou, and Vita, 2019"
    },
    6: {
        "file_name": "gmd-12-677-2019.pdf",
        "title": "GCAM v5.1: representing the linkages between energy, water, land, climate, and economic systems",
        "reference": "Calvin, K. et al. GCAM v5.1: representing the linkages between energy, water, land, climate, and economic systems. Geoscientific Model Development 12, 677–698 (2019).",
        "citation": "Calvin et al., 2019"
    },
    7: {
        "file_name": "Yu_2018_Environ._Res._Lett._13_034034.pdf",
        "title": "Implementing nationally determined contributions: building energy policies in India’s mitigation strategy",
        "reference": "Yu, S. et al. Implementing nationally determined contributions: building energy policies in India’s mitigation strategy. ENVIRONMENTAL RESEARCH LETTERS 13, (2018).",
        "citation": "Yu et al., 2018"
    },  
    8: {
        "file_name": "CCSP_Synthesis_and_Assessment_Product_21_Part_A_Sc.pdf",
        "title": None,
        "reference": "Clarke, L. et al. CCSP Synthesis and Assessment Product 2.1, Part A: Scenarios of Greenhouse Gas Emissions and Atmospheric Concentrations. https://www.researchgate.net/publication/266456421 (2006).",
        "citation": "Clarke et al., 2006"
    },
    9: {
        "file_name": None,
        "title": None,
        "reference": "Clarke, J. F. & Edmonds, J. A. Modelling energy technologies in a competitive market. Energy Economics 15, 123–129 (1993).",
        "citation": "Clarke and Edmonds, 1993"
    },
    10: {
        "file_name": None,
        "title": None,
        "reference": "Edmonds, J. et al. An Integrated Assessment of Climate Change and the Accelerated Introduction of Advanced Energy Technologies - An Application of MiniCAM 1.0. Mitigation and Adaptation Strategies for Global Change 1, 311–339 (1997).",
        "citation": "Edmonds et al., 1997"
    },
    11: {
        "file_name": None,
        "title": None,
        "reference": "Clarke, L. et al. Effects of long-term climate change on global building energy expenditures. Energy Economics 72, 667–677 (2018).",
        "citation": "Clarke et al., 2018"
    },
    12: {
        "file_name": None,
        "title": None,
        "reference": "Muratori, M. et al. Cost of power or power of cost: A US modeling perspective. RENEWABLE & SUSTAINABLE ENERGY REVIEWS 77, 861–874 (2017).",
        "citation": "Muatori et al., 2017"
    },
    13: {
        "file_name": None,
        "title": None,
        "reference": "Luckow, P., Wise, M. A., Dooley, J. J. & Kim, S. H. Large-scale utilization of biomass energy and carbon dioxide capture and storage in the transport and electricity sectors under stringent CO2 concentration limit scenarios. International Journal of Greenhouse Gas Control 4, 865–877 (2010).",
        "citation": "Luckow, Wise, Dooley, and Kim, 2010"
    },
    14: {
        "file_name": None,
        "title": None,
        "reference": "Edmonds, J. & Reilly, J. Global energy production and use to the year 2050. Energy 8, 419–432 (1983).",
        "citation": "Edmonds and Reilly, 1983"
    },
    
}


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_db():
    """Load vector database"""

    embeddings = OpenAIEmbeddings()

    db_dir = "./data/archive"

    return Chroma(persist_directory=db_dir, embedding_function=embeddings)


def result_to_df(docs):
    """Convert a similarity query result to a data frame."""

    d = {"source": [], "page_number": [], "reference": [], "text_excerpt": []}
    for i in docs:
        extracted_text = i.page_content.replace("\n", "")

        text_excerpt = f"...{extracted_text}..."

        metadata = i.metadata

        source_id = int(os.path.splitext(os.path.basename(metadata["source"]))[0])
        d["source"].append(st.session_state.citations[source_id]["citation"])
        d["page_number"].append(metadata["page"])
        d["reference"].append(st.session_state.citations[source_id]["reference"])
        d["text_excerpt"].append(text_excerpt)
        
    return pd.DataFrame(d)


def clear_text():
    """Clear text from entry"""

    st.session_state.user_input = ""


# load local css for formating
local_css("style.css")

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

if "response_tempate" not in st.session_state:
    st.session_state.response_tempate = """
    {}

    Sources:

    {}
    """

if "input" not in st.session_state:
    st.session_state["input"] = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "reference_list" not in st.session_state:
    st.session_state.reference_list = []

if "citation_list" not in st.session_state:
    st.session_state.citation_list = []

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
# search = st.container()

st.markdown("#### Search for relevant documents")

with st.expander("Click to Expand"):

    st.session_state.n_results = st.selectbox(
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
    st.session_state.query = st.text_input("Enter your query here:")

    # get similary results from vector db
    if len(st.session_state.query) > 0:

        docs = st.session_state.retriever.get_relevant_documents(st.session_state.query)

        # format query result
        st.session_state.query_result = result_to_df(docs)

    if st.session_state.query_result is not None:
        st.write(st.session_state.query_result)

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

st.markdown("#### Converse with the GCAM archive")

with st.expander("Click to Expand"):

    # qa = st.container()
    st.markdown("#### Conduct QA with the GCAM archive")

        # set up retriever
    st.session_state.retriever = st.session_state.db.as_retriever(
        search_type="similarity",
        # search_type="mmr",
        search_kwargs={
            "k": 8, 
            # "search_distance": 0.1
        }
    )

    if st.session_state.qa_chain is None and st.session_state.retriever is not None:

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                temperature=0.0,
                model_name="gpt-3.5-turbo",
                max_tokens=500
            ),
            retriever=st.session_state.retriever,
            return_source_documents=True
        )

    # enter chat here
    user_input = st.text_input(
        "Your input to the chat", 
        st.session_state.input, 
        key="input", 
        placeholder="What would you like to discuss?", 
        on_change=clear_text,
        label_visibility='hidden'
    )

    if len(st.session_state.chat_history) > -1 and user_input:

        with st.spinner(f"Generating a response..."):

            output = st.session_state.qa_chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

        source_references = []
        source_citations = []
        for source in output["source_documents"]:
            source_id = int(os.path.splitext(os.path.basename(source.metadata["source"]))[0])
            source_page = source.metadata["page"]
            citation = st.session_state.citations[source_id]["citation"]
            reference = st.session_state.citations[source_id]["reference"]

            formatted_citation = f"{citation} (Page {source_page})"

            # response specific
            if reference not in source_references:
                source_references.append(reference)

            if citation not in source_citations:
                source_citations.append(citation)

            # tally
            if reference not in st.session_state.reference_list:
                st.session_state.reference_list.append(reference)

            if citation not in st.session_state.citation_list:
                st.session_state.citation_list.append(citation)

        # format response
        formatted_response = st.session_state.response_tempate.format(output["answer"], "; ".join(source_citations))

        st.session_state.chat_history.append(
            (user_input, formatted_response)
        )

        for i, o in reversed(st.session_state.chat_history):
            st.info(i, icon="🧐")
            st.success(o, icon="🤖")



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






