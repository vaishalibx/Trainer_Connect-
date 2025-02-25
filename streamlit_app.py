import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer    #Embedding 
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from phi.assistant import Assistant
from phi.llm.groq import Groq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("_name_")

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    st.error("Supabase credentials are not set. Please check your environment variables.")
    st.stop()

try:
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error(f"Failed to create Supabase client: {e}")
    logger.error(f"Supabase client creation error: {e}", exc_info=True)
    st.stop()

@st.cache_resource
def get_assistant():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable.")
        st.stop()
    return Assistant(
        llm=Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key),
        description="I am a helpful AI assistant powered by Groq. How can I assist you today?",
    )

@st.cache_data
def load_data():
    try:
        response = supabase.table("persons").select("*").execute()
        data = response.data
        if not data:
            logger.warning("No data retrieved from Supabase")
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error loading data from Supabase: {e}", exc_info=True)
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data found in the Supabase 'persons' table. Please check your database connection and table contents.")
else:
    pass

@st.cache_resource
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    ids = []
    
    for _, row in data.iterrows():
        description = f"{row['name']}, {row['experience']} years of experience, skilled in {row['skills']}, and works in {row['domain']}."
        embedding = model.encode(description)
        ids.append(row['id'])
        embeddings.append(embedding)
    
    return ids, np.array(embeddings)

def search_similar(query_embedding, embeddings, top_k=3):
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    indices = np.argsort(similarities)[::-1][:top_k]
    return indices

if not df.empty:
    ids, embeddings = generate_embeddings(df)

st.title("Debuggers Connect AI")

user_input = st.text_input("What do you want to know?", "")

model = SentenceTransformer('all-MiniLM-L6-v2')
query_embedding = model.encode([user_input]) if user_input.strip() else None

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            try:
                response_generator = get_assistant().chat(user_input)
                response = "".join([chunk for chunk in response_generator if isinstance(chunk, str)])
                st.markdown(response)

                if query_embedding is not None and not df.empty:
                    indices = search_similar(query_embedding, embeddings)

                    st.markdown("### Related Trainers for your query:")
                    for idx in indices:
                        trainer = df.iloc[idx]
                        st.markdown(f"- *Name: {trainer['name']}, **Domain: {trainer['domain']}, **Skills: {trainer['skills']}, **Experience*: {trainer['experience']} Months")
                        
                        st.markdown(f'<a href="{trainer["links"]}" target="_blank"><button style="background-color:#4CAF50; color:white; padding:10px; border:none; border-radius:4px; cursor:pointer;">Chat with {trainer["name"]}</button></a>', unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                st.error(f"An error occurred while processing your query: {e}")
    else:
        st.warning("Please enter a question.")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)
