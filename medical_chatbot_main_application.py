import os
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
from vertexai.preview.generative_models import GenerativeModel
import vertexai
import gradio as gr
 
# === CONFIGURATION ===
MONGODB_URI = "" # MongoDB CLuster URL
SERVICE_ACCOUNT = "" # Service Account
PROJECT_ID = "" # Project ID from GCP
DB_NAME = "" # DB we created in MongoDB
COLLECTION_NAME = "" #Collection in the DB
EMBEDDING_DIM = 384  # For all-MiniLM-L6-v2


# === DEVICE SETUP ===
device = "cuda" if torch.cuda.is_available() else "cpu"


# === GEMINI SETUP ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT
vertexai.init(project=PROJECT_ID, location="us-central1")

# Gemini model for answering clinical questions based on transcription context.
gemini = GenerativeModel(
    "models/gemini-2.0-flash-lite-001",
    system_instruction="""
    You are a specialized medical AI assistant trained to analyze medical transcriptions.
    Your role is to extract relevant medical information and provide accurate, helpful responses based solely
    on the provided medical context. Always prioritize patient safety and medical accuracy.
    """
)
# Gemini model configuration for symptom-based diagnosis responses
gemini_diag = GenerativeModel(
    "models/gemini-2.0-flash-lite-001",
    system_instruction="""
    You are a warm and concise medical assistant. When given symptom descriptions and context, your job is to suggest the most likely diagnosis and the next clinical steps.
    Speak clearly and kindly like a helpful doctor. DO NOT reference records, documents, or phrases like "it's important to figure out what's going on" or "there could be a few possibilities."
    Instead, go straight to the point using a natural, reassuring tone.
    Example: "Hey! Based on what you're experiencing, it looks like you might be dealing with ____. A good next step would be _____."
    """
)


# === MODEL SETUP ===
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


# === MONGODB SETUP ===
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# === SEARCH + DIAGNOSIS ===
def suggest_diagnosis(symptom_description: str, top_k: int = 10) -> List[str]:
    query_vec = embed_model.encode(symptom_description, convert_to_tensor=True).tolist()
    pipeline = [
        {"$search": {"index": "rich_vec_index", "knnBeta": {"vector": query_vec, "path": "embedding", "k": top_k * 2}}},
        {"$project": {"transcription": 1, "parsed_entities": 1, "score": {"$meta": "searchScore"}}}
    ]
    raw_results = list(collection.aggregate(pipeline))
    pairs = [(symptom_description, doc["transcription"]) for doc in raw_results[:top_k]]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(rerank_scores, raw_results[:top_k]), key=lambda x: x[0], reverse=True)
 
    context = "\n---\n".join([doc["transcription"] for _, doc in reranked])
    prompt = f"""
    PATIENT SYMPTOMS:
    {symptom_description}
    
    SIMILAR CLINICAL CASES:
    {context}
    
    QUESTION:
    What is the likely diagnosis based on these symptoms and similar cases?
    """
    response = gemini_diag.generate_content(prompt)
    return response.text.strip()


# === GENERAL Q/A method to ask GEMINI ===
def ask_gemini(query, docs):
    if not docs:
        return "❌ No documents available for Gemini."
    context = "\n\n---\n\n".join([doc["transcription"] for doc in docs])
    prompt = f"""
    You are a clinical assistant. Below are anonymized clinical notes from patient records.
    Your job is to answer the user's question **using only the information contained in the notes**.
    DO NOT guess or refer to patients not mentioned in the notes.
    
    CLINICAL NOTES:
    {context}
    
    QUESTION:
    {query}
    
    INSTRUCTIONS:
    - Use only facts stated in the notes.
    - If insufficient info, respond: "Not enough information in the notes."
    """
    response = gemini.generate_content(prompt)
    return response.text.strip()


# === VECTOR SEARCH + RERANKING ===
def search_with_reranking(query, top_k=10):
    query_vec = embed_model.encode(query, convert_to_tensor=True).tolist()
    pipeline = [
        {"$search": {"index": "rich_vec_index", "knnBeta": {"vector": query_vec, "path": "embedding", "k": top_k * 2}}},
        {"$project": {
            "transcription": 1,
            "sample_name": 1,
            "description": 1,
            "medical_specialty": 1,
            "age": 1,
            "gender": 1,
            "parsed_entities": 1,
            "score": {"$meta": "searchScore"}
        }}
    ]
    raw_results = list(collection.aggregate(pipeline))
    pairs = [(query, doc["transcription"]) for doc in raw_results[:top_k]]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(rerank_scores, raw_results[:top_k]), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in reranked]


# Handles research-based queries by retrieving and summarizing relevant clinical records using Gemini
def research_ui(question):
    if not question.strip():
        return "Please enter a question.", None
    docs = search_with_reranking(question)
    answer = ask_gemini(question, docs)
    return "💬", answer


# Analyzes patient symptoms and suggests a probable diagnosis using similar clinical records and Gemini
def diagnose_ui(symptoms):
    if not symptoms.strip():
        return "Please enter symptoms.", None
    result = suggest_diagnosis(symptoms)
    return "🧠Here is the Diagnosis Explanation:", result 


# Updates chatbot with research-based response by calling research_ui and appending formatted messages to chat history
def handle_research(user_input, chat_history):
    header, response = research_ui(user_input)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": f"{header}\n\n{response}"})
    return "", chat_history


# Updates chatbot with diagnosis explanation by calling diagnose_ui and appending formatted messages to chat history
def handle_diagnose(user_input, chat_history):
    header, response = diagnose_ui(user_input)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": f"{header}\n\n{response}"})
    return "", chat_history


# User Interface code linked with appropriate methods.
with gr.Blocks(
    title="Clinical Assistant",
    theme=gr.themes.Base(),
    fill_height=True,
    css="""
    #custom-header-bar {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #e0f2f1;
        border-radius: 10px 10px 0 0;
        border-bottom: 1px solid #b2dfdb;
        padding: 10px 0 8px 0;
        margin-bottom: 0px;
    }
    .header-side {
        flex: 1;
        text-align: left;
        font-size: 1rem;
        color: #888;
        font-weight: 400;
        padding-left: 18px;
        letter-spacing: 0.5px;
    }
    .header-center {
        flex: 2;
        text-align: center;
        font-size: 2rem;
        # font-weight: bold;
        color: #222;
        letter-spacing: 0.5px;
    }
    .header-center-upper-font {
        font-weight: bold;
        font-family: 'Magneto', 'Brush Script MT', cursive, sans-serif;
    }
    .header-side.right {
        text-align: right;
        padding-left: 0;
        padding-right: 18px;
    }
    #sidebar_logo img {
        border-radius: 50%;
        object-fit: cover;
        height: 120px;
        width: 120px;
        display: block;
        margin: auto;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        background-color: #e0f6fa; 
    }
    div.styler.svelte-lngued {
        background-color: transparent !important;
        background: none !important;
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
        padding-bottom: 0 !important;
    }
    #button_row {
        padding-bottom: 1%;
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-top: 10px;
        background: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        outline: none !important;
    }
    body { background-color: #f2f7f9; }
    #chatbox {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.03);
        position: relative;
        overflow: hidden !important;
    }
    #sidebar { 
        background-color: #e0f6fa;
        padding: 20px; 
        border-radius: 12px; 
        font-family: 'Arial'; 
        color: #007791;
    }
    textarea {
        border-color: #007791;
        padding-right: 100px;
        padding-bottom: 20px;
        position: relative;
        resize: none;
    }
    .gr-textbox-container { position: relative; }
    button {
        background-color: #007791 !important;
        color: white !important;
        border-radius: 20px;
        font-size: 14px;
        padding: 6px 16px;
        white-space: nowrap;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    @media only screen and (max-width: 768px) {
        #layout { flex-direction: column; }
        #button_row { flex-direction: column; gap: 8px; justify-content: center; }
        #custom-header-bar { font-size: 1rem; }
        .header-center { font-size: 1.2rem; }
    }
    #chatbox_container button, #chatbox_container svg {
        display: none !important;
    }
    footer,
    footer * {
        display: none !important;
    }
    body {
        padding-bottom: 0px !important;
    }
    button[aria-label="Fullscreen"] {
        display: none !important;
    }
    """
) as iface:
    with gr.Row(elem_id="layout"):
        with gr.Column(scale=3, elem_id="chatbox"):
            # --- Custom Header (matches attached image) ---
            gr.HTML(
                '''
                <div id="custom-header-bar">
                    <span class="header-side">powered by MongoDB</span>
                    <span class="header-center">
                        <span class="header-center-upper-font">NaviDoc</span> <br>Clinical AI Assistant
                    </span>
                    <span class="header-side right">powered by GCP</span>
                </div>
                '''
            )
            with gr.Group():
                with gr.Row():
                    msg = gr.Textbox(placeholder="👋Welcome to NaviDoc - Clinical AI Assistant! How can I assist you today?", label="", lines=2, show_label=False, elem_id="message_input")
                with gr.Row(elem_id="button_row"):
                    diagnose_btn = gr.Button("🔍 Diagnose", elem_id="icon_button")
                    research_btn = gr.Button("🧠 Research", elem_id="submit_button")
                    clear_button = gr.Button("🧹 Clear Chat", elem_id="clear_button")
            chatbot = gr.Chatbot(type='messages', elem_id="chatbox_container", show_label=False, scale=1)
            gr.HTML("""
                    <script>
                    // Wait for DOM to load
                    window.addEventListener('DOMContentLoaded', function() {
                      // Observe changes to the chatbox container
                      const chatbox = document.querySelector('#chatbox_container');
                      if (chatbox) {
                        const observer = new MutationObserver(() => {
                        chatbox.scrollTop = chatbox.scrollHeight;
                        });
                      observer.observe(chatbox, { childList: true, subtree: true });
                      }
                    });
                    </script>

                    <link href="https://fonts.cdnfonts.com/css/navi" rel="stylesheet">
                    <link href="https://fonts.googleapis.com/css?family=Open+Sans:400,600&display=swap" rel="stylesheet">
                    """)

        with gr.Column(scale=1, elem_id="sidebar"):
            with gr.Row():
                gr.Image(value="chatbot_logo_1.png", label="", height=120, width=120, elem_id="sidebar_logo", show_download_button=False, show_share_button=False, container=False)
            gr.Markdown("""
                ### 🩺 Welcome to the Clinical Assistant
                This chatbot helps medical professionals retrieve accurate, evidence-based insights from existing clinical transcriptions.
                        
                **Diagnose:**
                 <br> Helps to diagnose the diseases or condition based on case record of the patient and provide initial provisional diagnosis. It can also suggest investigations that can be undertaken.
                
                **Research:**
                 <br> Provides medical information based on the health records stored.
                
                **Disclaimer:** <span style="font-size:1.2em;">🏥⚠️</span>
                 <br>
                    <span style="background-color:#0077911C; padding:6px 10px; border-radius:8px; display:inline-block;">
                        The responses are for informational purposes only and are not a substitute for professional medical advice, diagnosis, or treatment.
                    </span>
                
            """)

    diagnose_btn.click(handle_diagnose, [msg, chatbot], [msg, chatbot])
    research_btn.click(handle_research, [msg, chatbot], [msg, chatbot])
    clear_button.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    iface.launch(favicon_path="tab-favicon.svg", server_name="0.0.0.0", server_port=8080)

# iface.launch(favicon_path="tab-favicon.svg")