from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import streamlit as st

# Configure Google Generative AI API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)


@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        if pdf_reader.is_encrypted:
            try:
                pdf_reader.decrypt("")
            except Exception as e:
                st.error(f"Failed to decrypt PDF: {e}")
                return ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


@st.cache_resource
def load_or_create_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("Text chunks are empty. Cannot create vector store.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say "answer is not available in the context".

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, chain, vector_store):
    docs = vector_store.similarity_search(user_question)
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response['output_text']


def main():
    st.set_page_config(page_title="Chat with PDF", layout="centered", page_icon="📄")
    st.title("📄 Chat with Your PDF Document")

    # Initialize chat history and vector store in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for PDF upload
    st.sidebar.title("Upload PDF")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)

    if pdf_docs:
        if st.sidebar.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                if not text_chunks:
                    st.error("No text extracted from the PDFs. Please check the content.")
                else:
                    st.session_state.vector_store = load_or_create_vector_store(text_chunks)
                    st.success("PDF processed successfully!")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<footer style='text-align: center;'>Created by <span style='font-weight: 600;'>Ghulam Mahiyudin</span></footer>",
        unsafe_allow_html=True)

    # Create conversational chain
    chain = get_conversational_chain()

    # User question input
    user_question = st.text_input("Ask a Question:", placeholder="Type your question here...")

    if st.button("Get Response"):
        # Validate user input
        if not user_question:
            st.warning("Please enter a question before submitting.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload and process a PDF document before asking a question.")
        else:
            with st.spinner("Generating response..."):
                answer = user_input(user_question, chain, st.session_state.vector_store)
                # Add question and answer to chat history
                st.session_state.chat_history.insert(0, (user_question, answer))  # Insert at the top

    # Display chat history with styling
    if st.session_state.chat_history:
        for question, answer in st.session_state.chat_history:
            st.write("---")
            st.markdown(f"**👤 You:** _{question}_")
            st.markdown(f"**🤖 AI:** {answer}")

if __name__ == "__main__":
    main()
