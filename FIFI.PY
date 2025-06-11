# EventBot Pro - Version améliorée avec modularité, gestion d'erreurs et interface optimisée

# 🔹 Imports
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import gradio as gr
import sqlite3
import os

# 🔹 Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
DB_PATH = "./event_docs.db"
PDF_PATHS = [
    "pdfs/idees_themes.pdf",
    "pdfs/budget_mariage.pdf",
    "pdfs/logistique_conference.pdf",
    "pdfs/planning_evenement.pdf",
    "pdfs/to do list mariage.pdf"
]

# 🔹 Initialisation LLM et embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

# 🔹 Base de données SQLite
def get_db_connection():
    return sqlite3.connect(DB_PATH)

conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    type TEXT
)
""")
conn.commit()

# 🔹 Détection du type d'événement à partir du nom du fichier
def detect_type_from_filename(filename):
    fname = filename.lower()
    if "mariage" in fname:
        return "MARIAGE"
    elif "logistique" in fname or "conference" in fname:
        return "SEMINAIRE"
    elif "salon" in fname:
        return "SALON"
    elif "theme" in fname or "idee" in fname or "creatif" in fname or "ambiance" in fname:
        return "CRÉATIVITÉ"
    elif "budget" in fname or "finance" in fname:
        return "BUDGET"
    elif "planning" in fname or "programme" in fname:
        return "PLANIFICATION"
    else:
        return "SEMINAIRE"

# 🔹 Insertion des PDFs dans la base
def insert_pdfs():
    missing_files = []
    for pdf_path in PDF_PATHS:
        if not os.path.exists(pdf_path):
            missing_files.append(pdf_path)
            print(f"⚠️ Fichier manquant: {pdf_path}")
            continue
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            full_text = "\n".join(doc.page_content for doc in docs)
            title = os.path.basename(pdf_path)
            doc_type = detect_type_from_filename(title)
            cursor.execute("SELECT COUNT(*) FROM documents WHERE title = ?", (title,))
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO documents (title, content, type) VALUES (?, ?, ?)",
                           (title, full_text, doc_type))
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {pdf_path}: {str(e)}")
    
    conn.commit()
    if missing_files:
        print(f"⚠️ Fichiers manquants: {', '.join(missing_files)}")
        print("Veuillez vous assurer que tous les fichiers PDF sont présents dans le répertoire.")

insert_pdfs()

# 🔹 Chargement des documents depuis la base
def load_documents_from_db():
    cursor.execute("SELECT title, content, type FROM documents")
    rows = cursor.fetchall()
    return [Document(page_content=row[1], metadata={"title": row[0], "type": row[2]}) for row in rows]

# 🔹 Prompts spécialisés
prompt_map = {
    "BUDGET": PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es eventbot, expert en gestion financière d'événements.
Fais une estimation claire et stratégique selon le contexte.

📚 CONTEXTE:
{context}

❓ QUESTION:
{question}

💰 BUDGET:
"""
    ),
    "PLANIFICATION": PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es eventbot, expert en planification détaillée d'événements.

📚 CONTEXTE:
{context}

❓ QUESTION:
{question}

📅 PLAN D'ACTION:
"""
    ),
    "SEMINAIRE": PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es eventbot , spécialiste de l'organisation de séminaires professionnels.
Base-toi sur le contexte fourni pour répondre à la question suivante.

📚 CONTEXTE:
{context}

❓ QUESTION:
{question}

✅ RÉPONSE:
"""
    ),
    "MARIAGE": PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es eventbot, un expert de l'organisation de mariages inoubliables.

📚 CONTEXTE:
{context}

❓ QUESTION:
{question}

💖 CONSEIL:
"""
    ),
    "SALON": PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es eventbot, expert des salons et expositions.

📚 CONTEXTE:
{context}

❓ QUESTION:
{question}

🏢 STRATÉGIE:
"""
    ),
    "CRÉATIVITÉ": PromptTemplate(
        input_variables=["context", "question"],
        template="""
Tu es eventbot, un expert en idées thématiques et concepts d'ambiance.

📚 CONTEXTE:
{context}

❓ QUESTION:
{question}

✨ IDÉES:
"""
    )
}

# 🔹 Fonction pour créer le retriever à partir des documents
def create_retriever():
    all_docs = load_documents_from_db()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    if all_docs:
        chunks = splitter.split_documents(all_docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        dummy_doc = Document(page_content="Bienvenue sur EventBot Pro. Aucun document n'est encore disponible.",
                             metadata={"title": "Bienvenue", "type": "SEMINAIRE"})
        chunks = splitter.split_documents([dummy_doc])
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})

# 🔹 Initialisation du retriever et de la chaîne QA
retriever = create_retriever()
default_prompt = prompt_map["SEMINAIRE"]
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": default_prompt}
)

# 🔹 Classification simple du type d'événement
def detect_event_type(question):
    q = question.lower()
    if any(word in q for word in ["mariage", "noces", "fiançailles"]):
        return "MARIAGE"
    elif any(word in q for word in ["salon", "exposition", "stand"]):
        return "SALON"
    elif any(word in q for word in ["ambiance", "thème", "décoration", "idées", "créatif"]):
        return "CRÉATIVITÉ"
    elif any(word in q for word in ["budget", "financement", "coût", "prix"]):
        return "BUDGET"
    elif any(word in q for word in ["planning", "planification", "étapes", "programme"]):
        return "PLANIFICATION"
    else:
        return "SEMINAIRE"

# 🔹 Interface Gradio
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("""
    # 🤖 EventBot Pro - Assistant intelligent
    Posez votre question : thèmes, organisation, planning, mariage ou salon.
    """)

    chatbot = gr.Chatbot(label="🧠 Discussion", height=500, type="messages")
    question_input = gr.Textbox(label="Votre question", placeholder="Ex: Des idées de thèmes originaux pour un mariage en été?", lines=2)
    status_box = gr.Textbox(label="État", interactive=True)
    ask_btn = gr.Button("Envoyer")
    clear_btn = gr.Button("Effacer")

    def ask_question(question, chat_display):
        if not question.strip():
            return chat_display, "❗ Entrez une question."

        event_type = detect_event_type(question)
        qa_chain.combine_documents_chain.llm_chain.prompt = prompt_map.get(event_type, default_prompt)

        try:
            result = qa_chain.invoke({"query": question, "question": question})
            answer = result.get("result", "Je n'ai pas de réponse.")
        except Exception as e:
            print(f"Erreur : {e}")
            answer = "Erreur lors de la génération."

        chat_display.append({"role": "user", "content": question})
        chat_display.append({"role": "assistant", "content": answer})
        return chat_display, ""

    def clear_chat():
        return [], ""

    ask_btn.click(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, status_box])
    clear_btn.click(clear_chat, outputs=[chatbot, status_box])
    question_input.submit(ask_question, inputs=[question_input, chatbot], outputs=[chatbot, status_box])

# 🔹 Lancement de l'interface
if __name__ == "__main__":
    try:
        interface.launch(share=True)
    finally:
        conn.close()  # Ensure database connection is closed