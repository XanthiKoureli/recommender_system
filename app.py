import streamlit as st
from metapub import PubMedFetcher
from components.chat_utils import ChatAgent
from components.chat_prompts import chat_prompt_template, qa_template
from components.llm import llm
from components.layout_extensions import render_app_info
from backend.abstract_retrieval.pubmed_retriever import PubMedAbstractRetriever
from backend.data_repository.local_storage import LocalJSONStore
from backend.rag_pipeline.chromadb_rag import ChromaDbRag
from backend.rag_pipeline.embeddings import embeddings_function


# Instantiate objects
pubmed_client = PubMedAbstractRetriever(PubMedFetcher())
data_repository = LocalJSONStore(storage_folder_path="backend/data")
rag_client = ChromaDbRag(persist_directory="backend/chromadb_storage", embeddings=embeddings_function)
chat_agent = ChatAgent(prompt=chat_prompt_template, llm=llm)


def generate_question_from_notes(medical_notes: str) -> str:
    """Use the LLM to convert medical notes into a structured question."""
    prompt = f"Based on the following medical notes, generate a concise and relevant medical research question:\n\n{medical_notes}"
    generated_question = llm.invoke(prompt).content
    return generated_question


def main():
    st.set_page_config(
        page_title="Pubmed Abstract Chatbot",
        page_icon='💬',
        layout='wide'
    )

    # Define columns - this will make layout split horizontally
    column_logo, column_app_info, column_answer = st.columns([1, 4, 4])

    # In the second column, place text explaining the purpose of the app and some example scientific questions
    # that your user might ask.
    with (column_app_info):

        # Render app info including example questions as cues for the user
        render_app_info()

        st.markdown("""
        <style>
        .stForm {
            border: none !important;
            padding: 0px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Section to enter scientific question
        st.header("Enter your scientific question or upload medical notes!")
        placeholder_text = "Type your scientific question here..."

        with st.form(key="question_form", clear_on_submit=True):
            scientist_question = st.text_input("What is your question?", value="", placeholder=placeholder_text)
            
            # Add file uploader for medical notes
            uploaded_file = st.file_uploader("Or upload a .txt file with medical notes", type=["txt"])
            
            # This will be triggered by Enter key or button click
            get_articles = st.form_submit_button('Get articles & Answer')

        retrieved_abstracts = None
        medical_notes = None
        # Processing user question, fetching data
        with st.spinner('Fetching abstracts. This can take a while...'):
            
            if get_articles:
                if uploaded_file:
                    # Read medical notes from the uploaded file
                    medical_notes = uploaded_file.read().decode("utf-8")
                    # st.write(f"{medical_notes.split('.')[0]} ...")
                    retrieved_abstracts, query_simplified = pubmed_client.get_abstract_data(
                        medical_notes, input_is_medical_notes=True)

                elif scientist_question and scientist_question != placeholder_text:
                    # Process user question if no file is uploaded
                    retrieved_abstracts, query_simplified = pubmed_client.get_abstract_data(
                        scientist_question, input_is_medical_notes=False)

                else:
                    st.write('Please enter a question or upload a .txt file with medical notes.')
                # if scientist_question and scientist_question != placeholder_text:

                # Get abstracts data
                # retrieved_abstracts = pubmed_client.get_abstract_data(scientist_question)
                if retrieved_abstracts:
                    if medical_notes:
                        # Generate a proper question from medical notes
                        generated_question = generate_question_from_notes(medical_notes)
                        st.write(f"Generated question from medical notes: {generated_question}")
                    # Save abstracts to storage and create vector index
                    query_id = data_repository.save_dataset(
                        retrieved_abstracts,
                        scientist_question
                        if (scientist_question is not None and scientist_question != placeholder_text)
                        else generated_question,
                        query_simplified,
                        medical_notes if medical_notes else None)
                    documents = data_repository.create_document_list(retrieved_abstracts)
                    rag_client.create_vector_index_for_user_query(documents, query_id)

                    # Answer the user question and display the answer on the UI directly
                    vector_index = rag_client.get_vector_index_by_user_query(query_id)
                    retrieved_documents = chat_agent.retrieve_documents(
                        vector_index, scientist_question
                        if (scientist_question is not None and scientist_question != placeholder_text)
                        else generated_question)
                    chain = qa_template | llm

                    with column_answer:
                        # llm_answer = chain.invoke({
                        #     "question": scientist_question,
                        #     "retrieved_abstracts": retrieved_documents,
                        #     }).content
                        # st.session_state[scientist_question] = llm_answer
                        # data_repository.save_initial_answer(query_id, llm_answer)
                        # st.session_state.selected_query = scientist_question
                        if medical_notes:
                            # Generate a proper question from medical notes
                            question = generated_question
                        else:
                            question = scientist_question

                        # Use the generated question instead of raw medical notes
                        llm_answer = chain.invoke({
                            "question": question,
                            "retrieved_abstracts": retrieved_documents,
                        }).content

                        st.session_state[question] = llm_answer
                        data_repository.save_initial_answer(query_id, llm_answer)
                        st.session_state.selected_query = question
                        
                else:
                    st.write('No abstracts found.')

    # Beginning of the chatbot section
    # Display list of queries to select one to have a conversation about
    query_options = data_repository.get_list_of_queries()

    if query_options:
        st.header("Chat with the abstracts")
        selected_query = st.selectbox('Select a past query', options=list(query_options.values()),
                                      key='selected_query')

        # Initialize chat about some query from the history of user questions
        if selected_query:
            selected_query_id = next(key for key, val in query_options.items() if val == selected_query)
            vector_index = rag_client.get_vector_index_by_user_query(selected_query_id)

            st.write(data_repository.load_initial_answer(selected_query_id))

            # Clear chat history when switching query to chat about
            if 'prev_selected_query' in st.session_state and st.session_state.prev_selected_query != selected_query:
                chat_agent.reset_history()

            st.session_state.prev_selected_query = selected_query

            # Start chat session
            chat_agent.start_conversation(vector_index, selected_query)


if __name__ == "__main__":
    main()
