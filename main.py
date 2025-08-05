import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_perplexity import ChatPerplexity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# Page config
st.set_page_config(page_title="YT Transcript Chat", layout="wide",page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ3EBEHGZ4Q5nAUuY8zb6u-1O-CIsR4nOShhQ&s")

# Background + styling
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), url("https://img.freepik.com/premium-photo/red-blue-black-geometric-abstract-pattern-wallpaper_984013-7.jpg");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }

    /* Input field customization */
    input, textarea {
        caret-color: black !important; /* Typing cursor */
        outline: none !important;
        border: 2px solid black !important;
        box-shadow: none !important;
    }

    /* Streamlit input field styling */
    .stTextInput > div > div > input {
        background-color: #ffffffdd !important;
        color: black !important;
        border: 1px solid #aaa !important;
        border-radius: 6px !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #ffffffcc;
        color: black;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5em 2em;
        margin-top: 10px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Headings and text */
    .stMarkdown, .stTitle, .stSubheader {
        color: white;
    }
    /*Answer Box Styling*/
    .answer-box {
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        color: white;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = ChatPerplexity()

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def extract_the_website_id(url):
    match = re.match(r".*youtu\.be/([^?&]+)", url)
    if match:
        return match.group(1)
    match = re.match(r".*youtube\.com.*[?&]v=([^?&]+)", url)
    if match:
        return match.group(1)
    raise ValueError("Invalid or unsupported YouTube URL format.")

# Main layout with centered content
left, center, right = st.columns([1, 2, 1])
with center:
    st.title("  YouTube Video Q&A")
    st.markdown("Enter a YouTube video URL with only English captions and ask any question based on it.")

    video_url = st.text_input("🔗 YouTube Video URL:")
    question = st.text_input("❓ Ask a question based on the video:")

    # Centered button
    button_center = st.columns(3)
    with button_center[1]:
        get_answer_clicked = st.button("🚀 Get Answer", use_container_width=True)

    if get_answer_clicked:
        if not video_url or not question:
            st.warning("Please provide both a video URL and a question.")
        else:
            try:
                video_id = extract_the_website_id(video_url)
                transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=["en"])
                transcript = " ".join(chunk.text for chunk in transcript_list)
            except Exception:
                st.error("Transcript not available or invalid URL.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = splitter.create_documents([transcript])

            embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            vector_store = FAISS.from_documents(chunks, embedding)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }) | prompt | model | StrOutputParser()

            with st.spinner("Thinking..."):
                result = chain.invoke(question)
            st.markdown("### 💬 Answer:")
            st.markdown(
                f"""
                <div class="answer-box">
                    {result}
                </div>
                """,
                unsafe_allow_html=True
            )

