import streamlit as st

def load_custom_css():
    """Load all custom CSS styling for the chatbot"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {font-family: 'Inter', sans-serif;}
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f0f 100%);
        color: #e8e8e8;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        padding: 1.5rem 1rem;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #00d9ff !important;
        text-shadow: 0 0 15px rgba(0, 217, 255, 0.4);
        font-weight: 700;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #00d9ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 217, 255, 0.3);
        width: 100%;
        margin: 8px 0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 30px rgba(0, 217, 255, 0.6);
        background: linear-gradient(90deg, #00e5ff 0%, #00b8e6 100%);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background: linear-gradient(135deg, #378ecf 0%, #07547e 100%);
        border-left: 5px solid #0c54b0;
    }

    .user-message div:last-child {
        color: #000000 !important;
        font-weight: 500;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-left: 5px solid #48bb78;
        color: #e2e8f0;
    }
    
    .message-header {
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .user-message .message-header {
        color: #00d9ff;
    }
    
    .assistant-message .message-header {
        color: #48bb78;
    }

    .stTextInput input {
        background: rgb(10 144 168 / 50%);
        border: 2px solid #0ba4be;
        border-radius: 15px;
        color: #ffffff;
        padding: 1.5rem 1.2rem;
        font-size: 1.15rem;
        transition: all 0.3s;
    }

    .stTextInput input::placeholder {
        color: #ffffff;
        opacity: 0.8;
    }

    .stTextInput input:focus {
        border-color: #00e5ff;
        box-shadow: 0 0 25px rgba(0, 217, 255, 0.7);
        background: rgba(0, 217, 255, 0.6);
    }
    
    .stTextArea textarea {
        background: rgb(10 144 168 / 50%);
        border: 2px solid #0ba4be;
        border-radius: 15px;
        color: #ffffff;
        padding: 1.2rem;
        font-size: 1.05rem;
        transition: all 0.3s;
        min-height: 80px;
    }

    .stTextArea textarea::placeholder {
        color: #ffffff;
        opacity: 0.8;
    }

    .stTextArea textarea:focus {
        border-color: #00e5ff;
        box-shadow: 0 0 25px rgba(0, 217, 255, 0.7);
        background: rgba(0, 217, 255, 0.6);
    }
    
    .file-badge {
        display: inline-block;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 25px;
        margin: 6px 4px;
        font-size: 0.88rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        transition: all 0.3s;
    }
    
    .file-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
    
    h1 {
        color: #0f63bc !important;
        text-align: center;
        font-size: 3rem !important;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.6);
        margin-bottom: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.15) 0%, rgba(0, 153, 204, 0.15) 100%);
        border-left: 5px solid #00d9ff;
        border-radius: 12px;
        padding: 1rem;
        color: #e8e8e8;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.15) 0%, rgba(56, 161, 105, 0.15) 100%);
        border-left: 5px solid #48bb78;
        border-radius: 12px;
        color: #e8e8e8;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(237, 137, 54, 0.15) 0%, rgba(221, 107, 32, 0.15) 100%);
        border-left: 5px solid #ed8936;
        border-radius: 12px;
        color: #e8e8e8;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        color: #e8e8e8;
    }
    
    /* ====== ENHANCED EXPANDER STYLING ====== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.2) 0%, rgba(0, 153, 204, 0.2) 100%) !important;
        border-radius: 12px !important;
        border: 2px solid rgba(0, 217, 255, 0.3) !important;
        color: #00d9ff !important;
        font-weight: 600 !important;
        padding: 1rem 1.2rem !important;
        transition: all 0.3s ease !important;
        margin: 0.5rem 0 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.3) 0%, rgba(0, 153, 204, 0.3) 100%) !important;
        border-color: rgba(0, 217, 255, 0.6) !important;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3) !important;
        transform: translateY(-2px);
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%) !important;
        border: 2px solid rgba(0, 217, 255, 0.2) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        color: #e8e8e8 !important;
        margin-top: -10px !important;
    }
    
    .streamlit-expanderContent p {
        color: #e8e8e8 !important;
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
    }
    
    /* ====== DEBUG CHUNKS STYLING ====== */
    .debug-chunk-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 2px solid rgba(102, 126, 234, 0.4);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .debug-chunk-header {
        color: #667eea;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .chunk-metadata {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #00d9ff;
    }
    
    .chunk-content {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #48bb78;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        color: #e8e8e8;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    
    /* ====== SOURCE SECTIONS STYLING ====== */
    .sources-header {
        background: linear-gradient(90deg, rgba(72, 187, 120, 0.2) 0%, rgba(56, 161, 105, 0.2) 100%);
        border-left: 5px solid #48bb78;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0 1rem 0;
        color: #48bb78;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.2);
    }
    
    /* ====== JSON VIEWER STYLING ====== */
    .stJson {
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.85rem !important;
    }
    
    ::-webkit-scrollbar {width: 12px;}
    ::-webkit-scrollbar-track {background: #1a1a2e;}
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ffffff 0%, #e2e8f0 100%);
        border-radius: 10px;
        border: 2px solid #00d9ff;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 100%);
    }
    
    .stSpinner > div {
        border-top-color: #0ea5e9 !important;
    }
    
    [data-testid="stSidebar"] .stButton button {
        text-align: center;
        font-size: 0.95rem;
        padding: 0.8rem 1rem;
        text-transform: none;
        letter-spacing: 0;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] button[key="new_chat_btn"] {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        margin-bottom: 1rem !important;
        font-size: 1rem !important;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stat-card h3 {
        color: #00d9ff;
        margin-bottom: 0.5rem;
    }
    
    .stat-card p {
        color: #e8e8e8;
        font-size: 0.95rem;
    }
    
    .css-1kyxreq, [data-testid="stSidebar"] .caption, [data-testid="stSidebar"] caption {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebarNavButton"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebarNavButton"] svg {
        fill: #ffffff !important;
    }
    
    [data-testid="stSidebar"] small {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)