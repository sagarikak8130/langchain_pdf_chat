from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import uuid

from langchain_utils.pdf_processor import process_pdf
from langchain_utils.chain_builder import build_conversational_chain

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

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    chat_history = []

    if request.method == "POST":
        question = request.form.get("question")
        uploaded_file = request.files.get("pdf")

        if uploaded_file:
            # Save file
            filename = secure_filename(str(uuid.uuid4()) + "_" + uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            # Store file path in session
            session['filepath'] = filepath

            # Build vector store & chain
            vectorstore = process_pdf(filepath)
            chain, memory = build_conversational_chain(vectorstore)
            chains[session.sid] = {"chain": chain, "memory": memory}

        elif 'filepath' in session:
            filepath = session['filepath']

        # Answer question
        if question and session.sid in chains:
            chain = chains[session.sid]['chain']
            memory = chains[session.sid]['memory']
            answer = chain.invoke({"input": question, "chat_history": memory.chat_memory.messages})
            chat_history = memory.chat_memory.messages

    return render_template("index.html", answer=answer, chat_history=chat_history)
 
if __name__ == "__main__":
    app.run(debug=True)