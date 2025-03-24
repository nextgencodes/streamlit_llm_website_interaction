import streamlit as st
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import requests
import xml.etree.ElementTree as ET


# --- Setup ---
api_key = st.secrets["GOOGLE_API_KEY"] # Store your API key in Streamlit secrets for security
#api_key = st.text_input("Enter your Google AI Studio API Key:", type="password") # Password input

def initialize_llm(GOOGLE_API_KEY):
    """Initializes the Gemini Pro LLM."""
    return GoogleGenerativeAI(model="gemini-1.5-pro-001", google_api_key=GOOGLE_API_KEY)

def initialize_embeddings(GOOGLE_API_KEY):
    """Initializes the Gemini Pro embeddings model."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def ingest_urls(urls,GOOGLE_API_KEY):
    """Ingests content from URLs and creates a vector store."""
    loaders = [WebBaseLoader(url) for url in urls]
    documents = []
    for loader in loaders:
        try:  # Error handling for individual URL loading
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading URL: {loader.web_path}. Error: {e}")
    if not documents:
        return None # Return None if no documents loaded successfully

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(texts)
    embeddings = initialize_embeddings(GOOGLE_API_KEY)
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def create_qa_chain(vector_store, llm):
    """Creates a RetrievalQA chain."""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' is simple and good for smaller contexts
        retriever=vector_store.as_retriever()
    )
    return qa_chain

def fetch_sitemap_urls(url):
    """Fetches URLs from a sitemap.xml file."""
    sitemap_url = url.rstrip('/') + '/sitemap.xml' # Ensure no double slash
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        root = ET.fromstring(response.content)
        urls = []
        for element in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'): # Namespace is important!
            urls.append(element.text)
        return urls
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch sitemap from {sitemap_url}. Error: {e}")
        return [] # Return empty list if sitemap fetch fails
    except ET.ParseError as e:
        st.warning(f"Could not parse sitemap XML from {sitemap_url}. Error: {e}")
        return [] # Return empty list if sitemap parsing fails

# --- Streamlit App ---
st.title("Web Content Q&A Tool")

url_input = st.text_area("Enter URLs (one per line):", height=100)

col1, col2  = st.columns(2)

vector_store = None # Initialize vector_store outside button logic and session state is better for simple cases.
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

with col1:
    ingest_button1 = st.button("Ingest URLs", use_container_width=True)
with col2:
    ingest_button2 = st.button("Ingest all subdomains",use_container_width=True)

question = st.text_input("Ask a question about the content:")
ask_button = st.button("Ask Question")

if ingest_button1:
    urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()] # Split URLs and remove empty ones

    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        with st.spinner("Ingesting URLs..."):
            st.session_state['vector_store'] = ingest_urls(urls, api_key) # Store in session state

if ingest_button2:
    urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()] # Split URLs and remove empty ones

    if not urls:
        st.warning("Please enter at least one URL.")

    all_urls_to_load = []

    for url in urls:
        sitemap_urls = fetch_sitemap_urls(url)
        if sitemap_urls:
            all_urls_to_load.extend(sitemap_urls)
        else:
            all_urls_to_load.append(url) # If sitemap fails, fallback to original url

    if not all_urls_to_load:
        st.warning("No URLs to process after sitemap extraction.")

    else:
        with st.spinner("Ingesting URLs and subdomains..."):
            st.session_state['vector_store'] = ingest_urls(all_urls_to_load, api_key) # Store in session state


if ask_button:
    if st.session_state['vector_store'] is not None: # Check if vector_store exists
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Answering question..."):
                llm = initialize_llm(api_key)
                qa_chain = create_qa_chain(st.session_state['vector_store'], llm) # Retrieve from session state

                try:
                    response = qa_chain.run(question)
                    st.success("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error during question answering: {e}")
    else:
        st.warning("Please ingest URLs first before asking a question.")

