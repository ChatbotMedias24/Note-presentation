import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message  # Importez la fonction message
import toml
import docx2txt
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = []
st.markdown(
    """
    <style>

        .user-message {
            text-align: left;
            background-color: #E8F0FF;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: 10px;
            margin-right: -40px;
            color:black;
        }

        .assistant-message {
            text-align: left;
            background-color: #F0F0F0;
            padding: 8px;
            border-radius: 15px 15px 15px 0;
            margin: 4px 0;
            margin-left: -10px;
            margin-right: 10px;
            color:black;
        }

        .message-container {
            display: flex;
            align-items: center;
        }

        .message-avatar {
            font-size: 25px;
            margin-right: 20px;
            flex-shrink: 0; /* Empêcher l'avatar de rétrécir */
            display: inline-block;
            vertical-align: middle;
        }

        .message-content {
            flex-grow: 1; /* Permettre au message de prendre tout l'espace disponible */
            display: inline-block; /* Ajout de cette propriété */
}
        .message-container.user {
            justify-content: flex-end; /* Aligner à gauche pour l'utilisateur */
        }

        .message-container.assistant {
            justify-content: flex-start; /* Aligner à droite pour l'assistant */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar contents
textcontainer = st.container()
with textcontainer:
    logo_path = "medi.png"
    logoo_path = "NOTEPRESENTATION.png"
    st.sidebar.image(logo_path,width=150)
   
    
st.sidebar.subheader("Suggestions:")
questions = [
        "Donnez-moi un résumé du rapport ",
        "Quelles sont les actions prévues pour améliorer l'accès à l'éducation et aux services de santé dans le PLF 2025 ?",
        "Quels sont les défis majeurs auxquels fait face l'élaboration du PLF 2025 ?",
        "Quels sont les objectifs du gouvernement en matière de développement du dessalement de l’eau de mer ?",
        "Quels sont les principaux secteurs prioritaires dans le budget 2025 du Maroc ?"

    ]    
 
load_dotenv(st.secrets["OPENAI_API_KEY"])
conversation_history = StreamlitChatMessageHistory()

def main():
    conversation_history = StreamlitChatMessageHistory()  # Créez l'instance pour l'historique
    st.header("Rapport sur le Projet de Loi de Finances pour l’année budgétaire 2025.  💬")
    
    # Load the document
    docx = 'note présentation (2).docx'
    
    if docx is not None:
        text = docx2txt.process(docx)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open("aaa.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

        st.markdown('<p style="margin-bottom: 0;"><h7><b>Posez vos questions ci-dessous:</b></h7></p>', unsafe_allow_html=True)

        query_input = st.text_input("")
        selected_questions = st.sidebar.radio("****Choisir :****", questions)
        
        # Initialize query
        query = ""
        
        if query_input and query_input not in st.session_state.previous_question:
            query = query_input
            st.session_state.previous_question.append(query_input)
        elif selected_questions:
            query = selected_questions
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                
                if "Donnez-moi un résumé du rapport" in query:
                    response = "Le Projet de Loi de Finances (PLF) 2025 du Maroc présente la répartition des dépenses prévues par ministère et institution, en mettant l'accent sur les secteurs prioritaires tels que l'éducation, la santé, et l'infrastructure. Les budgets les plus importants sont alloués au Ministère de l'Éducation Nationale (87,6 milliards de dirhams), au Ministère de la Santé (32,5 milliards), et au Ministère de l'Économie et des Finances (91,7 milliards). Des fonds spéciaux sont également prévus pour le développement régional, la gestion des catastrophes, et la promotion de l'emploi. L'Administration de la Défense Nationale bénéficie d'un budget substantiel de 58,7 milliards de dirhams pour soutenir les Forces Armées Royales et leurs opérations. Le projet reflète les priorités du pays en matière de développement durable et d'amélioration des services publics."

                conversation_history.add_user_message(query)
                conversation_history.add_ai_message(response)

            formatted_messages = []
            for msg in conversation_history.messages:
                role = "user" if msg.type == "human" else "assistant"
                avatar = "🧑" if role == "user" else "🤖"
                css_class = "user-message" if role == "user" else "assistant-message"
                
                message_div = f'<div class="{css_class}">{msg.content}</div>'
                avatar_div = f'<div class="avatar">{avatar}</div>'
                
                if role == "user":
                    formatted_message = f'<div class="message-container user"><div class="message-avatar">{avatar_div}</div><div class="message-content">{message_div}</div></div>'
                else:
                    formatted_message = f'<div class="message-container assistant"><div class="message-content">{message_div}</div><div class="message-avatar">{avatar_div}</div></div>'
                
                formatted_messages.append(formatted_message)

            messages_html = "\n".join(formatted_messages)
            st.markdown(messages_html, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
