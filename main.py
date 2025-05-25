import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "rag-chatbot")

# Initialize Pinecone and Gemini
@st.cache_resource
def initialize_services():
    """Initialize Pinecone and Gemini services"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        
        # Initialize Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        
        return pc, index
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        return None, None

def get_query_embedding(query):
    """Get embedding for user query"""
    try:
        model = 'models/text-embedding-004'
        response = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query"
        )
        return response['embedding']
    except Exception as e:
        st.error(f"Error getting query embedding: {e}")
        return None

def search_similar_chunks(index, query_embedding, top_k=5):
    """Search for similar chunks in Pinecone"""
    try:
        search_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        return search_response['matches']
    except Exception as e:
        st.error(f"Error searching Pinecone: {e}")
        return []

def generate_response(query, context_chunks):
    """Generate response using Gemini with retrieved context"""
    try:
        # Prepare context from retrieved chunks
        context = "\n\n".join([match['metadata']['text'] for match in context_chunks])
        
        # Create prompt
        prompt = f"""You are Soham's personal AI assistant. Answer questions about Soham using the provided context.
        Be friendly, helpful, and speak as if you know Soham personally. If the answer is not in the context, 
        politely say "I don't have that specific information about Soham right now."

Context:
{context}

Question: {query}

Answer:"""

        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating the response."

def main():
    # Page configuration
    st.set_page_config(
        page_title="Soham's Personal Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for dark animated theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding-top: 1rem;
    }
    
    .stApp {
        background: #0a0a0a;
        overflow-x: hidden;
    }
    
    /* Animated Star Background */
    .stars-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh;
        z-index: -2;
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 100%);
    }
    
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        animation: twinkle 2s linear infinite;
    }
    
    .star:nth-child(1) { width: 1px; height: 1px; top: 20%; left: 20%; animation-delay: 0s; }
    .star:nth-child(2) { width: 2px; height: 2px; top: 60%; left: 30%; animation-delay: 0.5s; }
    .star:nth-child(3) { width: 1px; height: 1px; top: 10%; left: 70%; animation-delay: 1s; }
    .star:nth-child(4) { width: 2px; height: 2px; top: 80%; left: 80%; animation-delay: 1.5s; }
    .star:nth-child(5) { width: 1px; height: 1px; top: 40%; left: 10%; animation-delay: 2s; }
    .star:nth-child(6) { width: 2px; height: 2px; top: 30%; left: 90%; animation-delay: 0.3s; }
    .star:nth-child(7) { width: 1px; height: 1px; top: 70%; left: 50%; animation-delay: 1.2s; }
    .star:nth-child(8) { width: 2px; height: 2px; top: 15%; left: 40%; animation-delay: 0.8s; }
    .star:nth-child(9) { width: 1px; height: 1px; top: 90%; left: 20%; animation-delay: 1.8s; }
    .star:nth-child(10) { width: 2px; height: 2px; top: 50%; left: 60%; animation-delay: 0.2s; }
    
    .shooting-star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: linear-gradient(45deg, #ffffff, #64ffda);
        border-radius: 50%;
        box-shadow: 0 0 10px #64ffda;
        animation: shoot 3s linear infinite;
    }
    
    .shooting-star:nth-child(11) { top: 10%; left: -5%; animation-delay: 0s; }
    .shooting-star:nth-child(12) { top: 30%; left: -5%; animation-delay: 2s; }
    .shooting-star:nth-child(13) { top: 70%; left: -5%; animation-delay: 4s; }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    
    @keyframes shoot {
        0% { transform: translateX(-100px) translateY(-100px); opacity: 1; }
        100% { transform: translateX(100vw) translateY(100vh); opacity: 0; }
    }
    
    /* Cosmic Particles */
    .particle {
        position: absolute;
        background: radial-gradient(circle, #64ffda, transparent);
        border-radius: 50%;
        animation: float 8s ease-in-out infinite;
    }
    
    .particle:nth-child(14) { width: 4px; height: 4px; top: 25%; left: 15%; animation-delay: 0s; }
    .particle:nth-child(15) { width: 3px; height: 3px; top: 75%; left: 85%; animation-delay: 2s; }
    .particle:nth-child(16) { width: 5px; height: 5px; top: 45%; left: 25%; animation-delay: 4s; }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) scale(1); opacity: 0.7; }
        50% { transform: translateY(-20px) scale(1.1); opacity: 1; }
    }
    
    /* Main Header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .main-title {
        font-size: 3.8rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(100, 255, 218, 0.5), 0 0 40px rgba(100, 255, 218, 0.3);
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #64ffda, #bb86fc, #03dac6, #ffffff);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease-in-out infinite;
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: #bb86fc;
        font-weight: 400;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(187, 134, 252, 0.5);
    }
    
    .robot-emoji {
        font-size: 4.5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 20px rgba(100, 255, 218, 0.6));
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Welcome Card */
    .welcome-card {
        background: rgba(18, 18, 18, 0.9);
        border: 1px solid rgba(100, 255, 218, 0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 700px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .welcome-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(100, 255, 218, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .welcome-text {
        font-size: 1.3rem;
        color: #e0e0e0;
        text-align: center;
        line-height: 1.8;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .welcome-text strong {
        color: #64ffda;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
    }
    
    /* Chat Container */
    .chat-container {
        background: rgba(18, 18, 18, 0.8);
        border: 1px solid rgba(187, 134, 252, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem auto;
        max-width: 900px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: transparent !important;
        border-radius: 15px;
        margin: 1rem 0;
        border: none !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #bb86fc, #6200ea) !important;
        border-radius: 20px 20px 5px 20px !important;
        padding: 1rem 1.5rem !important;
        margin-left: 2rem !important;
        box-shadow: 0 4px 15px rgba(187, 134, 252, 0.3) !important;
        border: 1px solid rgba(187, 134, 252, 0.5) !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, #1e1e1e, #2d2d2d) !important;
        border-radius: 20px 20px 20px 5px !important;
        padding: 1rem 1.5rem !important;
        margin-right: 2rem !important;
        box-shadow: 0 4px 15px rgba(100, 255, 218, 0.2) !important;
        border: 1px solid rgba(100, 255, 218, 0.3) !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] p {
        color: #e0e0e0 !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
    }
    
    /* Chat Input */
    .stChatInput > div {
        background: rgba(30, 30, 30, 0.9) !important;
        border: 2px solid rgba(100, 255, 218, 0.3) !important;
        border-radius: 25px !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #64ffda !important;
        box-shadow: 0 0 20px rgba(100, 255, 218, 0.3) !important;
    }
    
    .stChatInput input {
        color: #ffffff !important;
        background: transparent !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
    }
    
    .stChatInput input::placeholder {
        color: #888888 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #64ffda !important;
        border-right-color: #bb86fc !important;
    }
    
    /* Error Messages */
    .stError {
        background: rgba(244, 67, 54, 0.1) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        color: #ff6b6b !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(100, 255, 218, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 255, 218, 0.5);
    }
    
    /* Hide Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    header { visibility: hidden; }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
        }
        
        .welcome-card {
            margin: 1rem;
            padding: 1.5rem;
        }
        
        .chat-container {
            margin: 0.5rem;
            padding: 1rem;
        }
        
        .stChatMessage[data-testid="chat-message-user"],
        .stChatMessage[data-testid="chat-message-assistant"] {
            margin-left: 0.5rem !important;
            margin-right: 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Animated background with stars
    st.markdown("""
    <div class="stars-container">
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="star"></div>
        <div class="shooting-star"></div>
        <div class="shooting-star"></div>
        <div class="shooting-star"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <div class="robot-emoji">🤖</div>
        <h1 class="main-title">Soham's Personal Assistant</h1>
        <p class="subtitle">⚡ AI-Powered • 🌟 Always Ready • 🚀 Ultra Smart</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize services
    pc, index = initialize_services()
    
    if not pc or not index:
        st.error("🚨 Unable to connect to AI services. Please check configuration.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Welcome message for first-time users
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-text">
                👋 <strong>Hey! I'm Soham's Personal Assistant</strong><br><br>
                🌟 You can ask me anything about Soham - his background, interests, projects, or experiences.<br><br>
                🎯 I'm powered by advanced AI and ready to provide you with accurate and helpful information!<br><br>
                💫 Go ahead, start a conversation...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat container
    if st.session_state.messages:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("💭 Ask me anything about Soham..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Create chat container if this is the first message
        if len(st.session_state.messages) == 1:
            st.rerun()
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question..."):
                # Get query embedding
                query_embedding = get_query_embedding(prompt)
                
                if query_embedding:
                    # Search for similar chunks
                    similar_chunks = search_similar_chunks(index, query_embedding, top_k=5)
                    
                    if similar_chunks:
                        # Generate response
                        response = generate_response(prompt, similar_chunks)
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        error_msg = "I don't have that specific information about Soham right now. Feel free to ask me something else! ✨"
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = "I'm having trouble processing your question. Could you please try rephrasing it? 🤔"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
