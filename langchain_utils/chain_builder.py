from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)

def build_conversational_chain(vectorstore):
    logger.info("Initializing conversational chain")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    logger.info("Memory initialized")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    logger.info("LLM initialized with gpt-3.5-turbo")

    condense_question_system_template = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    condense_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
    )
    logger.info("Question condensation prompt template created")

    history_aware_retriever  = create_history_aware_retriever(
        llm,
        vectorstore.as_retriever(),
        condense_question_prompt
    )
    logger.info("History aware retriever created")
    
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    qa_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    logger.info("QA prompt template created")

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    logger.info("QA chain created")

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    logger.info("Conversational QA chain created successfully")

    return convo_qa_chain, memory
