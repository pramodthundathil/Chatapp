import logging
import sys
import torch
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import  ServiceContext
from llama_index.embeddings import LangchainEmbedding


import streamlit as st
from streamlit_chat import message

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("Data").load_data()

st.title("Legal Chat Bot")
st.title("Personalized Advocate")
st.markdown('<style>h1{color: Black; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Get Your Legal Openion')
st.markdown('<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

custom_cache_folder = "C:/Users/Digital/.cache\huggingface/hub/models--thenlper--gte-large/snapshots/8cb729e8b44d9ec9d85c1cec4167ed28b43b04c2/1_Pooling/"
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="thenlper/gte-large", cache_folder=custom_cache_folder)
)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=llm,
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)


def QueryAsk(qury):
    query_engine = index.as_query_engine()
    response = query_engine.query(qury)
    return response



if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! I am your Legal Assistent"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query):
    result = QueryAsk( query)
    st.session_state['history'].append((query, result))
    return result

reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Ask about your Job Interview", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
