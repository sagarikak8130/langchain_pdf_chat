from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import uuid
import logging

from langchain_utils.pdf_processor import process_pdf
from langchain_utils.chain_builder import build_conversational_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store retriever & memory per session
chains = {}

@app.before_request
def assign_session_id():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    chat_history = []
    user_id = session['user_id']

    if request.method == "POST":
        question = request.form.get("question")
        uploaded_file = request.files.get("pdf")

        if uploaded_file:
            logger.info(f"Processing new PDF upload: {uploaded_file.filename}")
            # Save file
            filename = secure_filename(str(uuid.uuid4()) + "_" + uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)
            logger.info(f"PDF saved to: {filepath}")

            # Store file path in session
            session['filepath'] = filepath

            # Build vector store & chain
            logger.info("Building vector store and chain")
            vectorstore = process_pdf(filepath)
            chain, memory = build_conversational_chain(vectorstore)
            chains[user_id] = {"chain": chain, "memory": memory}
            logger.info("Vector store and chain built successfully")

        elif 'filepath' in session:
            filepath = session['filepath']
            logger.info(f"Using existing PDF: {filepath}")

        # Answer question
        if question and user_id in chains:
            logger.info(f"Processing question: {question}")
            chain = chains[user_id]['chain']
            memory = chains[user_id]['memory']
            # answer = chain.invoke({"input": question, "chat_history": memory.chat_memory.messages})
            response = chain.invoke({"input": question,"chat_history": memory.chat_memory.messages})
            answer = response['answer']
            chat_history = memory.chat_memory.messages
            logger.info("Question processed successfully")

    return render_template("index.html", answer=answer, chat_history=chat_history)
 
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)