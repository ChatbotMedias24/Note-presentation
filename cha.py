import streamlit as st
import openai
import streamlit as st
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
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
import docx2txt
from dotenv import load_dotenv
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = []

# Chargement de l'API Key depuis les variables d'environnement
load_dotenv(st.secrets["OPENAI_API_KEY"])

# Configuration de l'historique de la conversation
if 'previous_questions' not in st.session_state:
    st.session_state.previous_questions = []

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
        input[type="text"] {
            background-color: #E0E0E0;
        }

        /* Style for placeholder text with bold font */
        input::placeholder {
            color: #555555; /* Gris foncé */
            font-weight: bold; /* Mettre en gras */
        }

        /* Ajouter de l'espace en blanc sous le champ de saisie */
        .input-space {
            height: 20px;
            background-color: white;
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
# Initialisation de l'historique de la conversation dans `st.session_state`
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = StreamlitChatMessageHistory()
def main():
    conversation_history = StreamlitChatMessageHistory()  # Créez l'instance pour l'historique

    st.header("PLF2025: Explorez le rapport sur note de présentation à travers notre chatbot 💬")
    
    # Load the document
    docx = 'PLF2025- Note de présentation_Fr-1-160.docx'
    
    if docx is not None:
        # Lire le texte du document
        text = docx2txt.process(docx)

        # Afficher toujours la barre de saisie
        st.markdown('<div class="input-space"></div>', unsafe_allow_html=True)
        selected_questions = st.sidebar.radio("****Choisir :****", questions)
        # Afficher toujours la barre de saisie
        query_input = st.text_input("", key="text_input_query", placeholder="Posez votre question ici...", help="Posez votre question ici...")
        st.markdown('<div class="input-space"></div>', unsafe_allow_html=True)

        if query_input and query_input not in st.session_state.previous_question:
            query = query_input
            st.session_state.previous_question.append(query_input)
        elif selected_questions:
            query = selected_questions
        else:
            query = ""

        if query :
            st.session_state.conversation_history.add_user_message(query) 
            if "Donnez-moi un résumé du rapport" in query:
                summary="Le "Projet de Loi de Finances 2025" se concentre sur trois priorités principales : la relance économique post-pandémie, la poursuite des réformes structurelles et le renforcement des politiques sociales. Il vise à stimuler la croissance économique tout en réduisant le déficit budgétaire et en augmentant les investissements publics. Des mesures spécifiques sont prévues pour soutenir les secteurs clés tels que l'agriculture, l'industrie et les énergies renouvelables. Par ailleurs, le projet prévoit des réformes fiscales et des initiatives pour améliorer la compétitivité des entreprises, tout en accordant une attention particulière à la protection des couches sociales les plus vulnérables."
                st.session_state.conversation_history.add_ai_message(summary) 

            else:
                messages = [
                {
                    "role": "user",
                    "content": (
                        f"{query}. En tenant compte du texte suivant, merci de formuler une réponse en évitant de mentionner l'absence d'informations, même si certaines données manquent. Répondez en vous appuyant sur vos connaissances et évitez de signaler que le texte est incomplet ou fragmenté. L'objectif est de fournir une réponse claire et complète, sans critique du texte, car elle sera directement affichée au lecteur.essayer de repondre à partir de texte {text} "
                    )
                }
            ]

            # Appeler l'API OpenAI pour obtenir le résumé
                response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages
            )

            # Récupérer le contenu de la réponse

                summary = response['choices'][0]['message']['content']
           
                # Votre logique pour traiter les réponses
            #conversation_history.add_user_message(query)
            #conversation_history.add_ai_message(response)
                st.session_state.conversation_history.add_ai_message(summary)  # Ajouter à l'historique
            
            # Afficher la question et le résumé de l'assistant
            #conversation_history.add_user_message(query)
            #conversation_history.add_ai_message(summary)

            # Format et afficher les messages comme précédemment
                
            # Format et afficher les messages comme précédemment
            formatted_messages = []
            previous_role = None 
            if st.session_state.conversation_history.messages: # Variable pour stocker le rôle du message précédent
                    for msg in conversation_history.messages:
                        role = "user" if msg.type == "human" else "assistant"
                        avatar = "🧑" if role == "user" else "🤖"
                        css_class = "user-message" if role == "user" else "assistant-message"

                        if role == "user" and previous_role == "assistant":
                            message_div = f'<div class="{css_class}" style="margin-top: 25px;">{msg.content}</div>'
                        else:
                            message_div = f'<div class="{css_class}">{msg.content}</div>'

                        avatar_div = f'<div class="avatar">{avatar}</div>'
                
                        if role == "user":
                            formatted_message = f'<div class="message-container user"><div class="message-avatar">{avatar_div}</div><div class="message-content">{message_div}</div></div>'
                        else:
                            formatted_message = f'<div class="message-container assistant"><div class="message-content">{message_div}</div><div class="message-avatar">{avatar_div}</div></div>'
                
                        formatted_messages.append(formatted_message)
                        previous_role = role  # Mettre à jour le rôle du message précédent

                    messages_html = "\n".join(formatted_messages)
                    st.markdown(messages_html, unsafe_allow_html=True)
if __name__ == '__main__':
    main()

