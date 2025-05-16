import streamlit as st
from cerebras_client import get_chat_completion, get_text_completion, DEFAULT_MODEL

def render_cerebras_tab():
    """Render the Cerebras LLM Completion tab."""
    st.header("Cerebras LLM Completion")

    with st.form(key="cerebras_completion_form", clear_on_submit=False):
        st.write("Interact with Cerebras Large Language Models.")
        
        selected_completion_type = st.radio(
            "Select completion type:",
            options=["Chat Completion", "Text Completion"],
            horizontal=True
        )

        selected_model = st.selectbox(
            "Select model:",
            options=[
                "qwen-3-32b",
                "llama-4-scout-17b-16e-instruct",
                "llama3.1-8b",
                "llama-3.3-70b"
            ],
            index=0
        )

        user_prompt_input = st.text_area(
            "Enter your prompt:",
            height=150,
            placeholder="e.g., Explain the theory of relativity in simple terms."
        )

        form_submitted = st.form_submit_button("Generate Completion")

        if form_submitted:
            if user_prompt_input.strip():
                with st.spinner("Generating completion from Cerebras LLM..."):
                    try:
                        api_response = None
                        if selected_completion_type == "Chat Completion":
                            st.info("Requesting Chat Completion...")
                            api_response = get_chat_completion(user_prompt_input, model=selected_model)
                        else:
                            st.info("Requesting Text Completion...")
                            api_response = get_text_completion(user_prompt_input, model=selected_model)

                        st.subheader("LLM Response:")
                        
                        # Debug information
                        with st.expander("Response Type Info"):
                            st.write(f"Response type: {type(api_response)}")
                        
                        # Try attribute access method first
                        try:
                            if selected_completion_type == "Chat Completion":
                                # Rest of your implementation...
                                response_text = api_response.choices[0].message.content
                                st.markdown(response_text)
                                
                                with st.expander("Response Metadata"):
                                    st.write(f"**Finish Reason:** {api_response.choices[0].finish_reason}")
                                    st.write(f"**Model:** {api_response.model}")
                                    if hasattr(api_response, 'usage'):
                                        st.write(f"**Usage:** {api_response.usage}")
                            
                            elif selected_completion_type == "Text Completion":
                                # Rest of your implementation...
                                try:
                                    response_text = api_response.choices[0].text
                                    st.markdown(response_text)
                                    
                                    with st.expander("Response Metadata"):
                                        st.write(f"**Finish Reason:** {api_response.choices[0].finish_reason}")
                                        st.write(f"**Model:** {api_response.model}")
                                        if hasattr(api_response, 'usage'):
                                            st.write(f"**Usage:** {api_response.usage}")
                                except AttributeError:
                                    # Your fallback implementation...
                                    if isinstance(api_response, dict) and "choices" in api_response and len(api_response["choices"]) > 0:
                                        if "text" in api_response["choices"][0]:
                                            response_text = api_response["choices"][0]["text"]
                                            st.markdown(response_text)
                                        else:
                                            st.error("Could not find 'text' in the completion response")
                                            st.json(api_response)
                                    else:
                                        st.error("Unexpected text completion response format")
                                        st.json(api_response)
                        
                        except (AttributeError, IndexError, TypeError) as attr_error:
                            # Your fallback implementation...
                            st.warning(f"Could not access response with attribute notation: {attr_error}")
                            
                            # Fall back to dictionary-based access
                            response_text = None
                            
                            # Your dictionary-based parsing logic...
                            if isinstance(api_response, dict):
                                # Rest of your implementation...
                                # This is abbreviated for brevity...
                                pass
                            
                            if response_text:
                                st.markdown(response_text)
                            else:
                                st.error("Could not extract response text using any known method")
                                st.json(api_response)

                    except Exception as e_cerebras:
                        st.error(f"An error occurred with the Cerebras API: {e_cerebras}")
            else:
                st.warning("Please enter a prompt before submitting.")