import os
import streamlit as st
from cerebras_client import get_chat_completion, get_text_completion

st.title("Cerebras Completion App")

# wrap in a form so ENTER key triggers submission
with st.form(key="completion_form", clear_on_submit=False):
    # Add radio button for selecting completion type
    completion_type = st.radio(
        "Select completion type:",
        options=["Chat Completion", "Text Completion"],
        horizontal=True
    )
    
    user_input = st.text_input("Enter your prompt:")
    submitted = st.form_submit_button("Submit")

    if submitted:
        if user_input:
            # Call the appropriate completion function based on user selection
            if completion_type == "Chat Completion":
                response = get_chat_completion(user_input)
                st.write("Chat Response:", response)
            else:  # Text Completion
                response = get_text_completion(user_input)
                st.write("Text Response:", response)
        else:
            st.write("Please enter a prompt.")