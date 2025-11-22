"""Buddy (Chat) Page"""
import streamlit as st
import os
from utils.db import SimpleDB
from utils.llm import SimpleLLM
import time


def render():
    st.title("💬 Buddy - AI Chat")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("API Key not configured")
        st.info("Go to Settings to configure your Google Gemini API key")
        return
    
    db = SimpleDB()
    user_id = st.session_state.get("user_id", "user_001")
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Message counter for triggering scroll
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    
    current_message_count = len(st.session_state.messages)
    
    # Display chat messages
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Auto-scroll to bottom when new messages arrive
    if current_message_count > st.session_state.message_count:
        # Inject JavaScript to scroll to bottom
        st.markdown(
            f"""
            <script>
                // Function to scroll to bottom
                function scrollToBottom() {{
                    // Try multiple scroll methods for compatibility
                    
                    // Method 1: Scroll main content area
                    const mainContent = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                    if (mainContent) {{
                        mainContent.scrollTop = mainContent.scrollHeight;
                    }}
                    
                    // Method 2: Scroll chat input container
                    const chatInput = window.parent.document.querySelector('[data-testid="stChatInput"]');
                    if (chatInput) {{
                        chatInput.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
                    }}
                    
                    // Method 3: Find last chat message and scroll to it
                    const chatMessages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                    if (chatMessages.length > 0) {{
                        const lastMessage = chatMessages[chatMessages.length - 1];
                        lastMessage.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                    }}
                    
                    // Method 4: Scroll window to bottom
                    window.parent.scrollTo({{
                        top: window.parent.document.documentElement.scrollHeight,
                        behavior: 'smooth'
                    }});
                }}
                
                // Execute scroll immediately
                scrollToBottom();
                
                // Also execute after short delays to catch any delayed rendering
                setTimeout(scrollToBottom, 50);
                setTimeout(scrollToBottom, 150);
                setTimeout(scrollToBottom, 300);
            </script>
            <div id="scroll-anchor-{current_message_count}"></div>
            """,
            unsafe_allow_html=True
        )
        st.session_state.message_count = current_message_count
    
    # Chat input
    if prompt := st.chat_input("Say something..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        db.add_chat(user_id, "user", prompt)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        try:
            llm = SimpleLLM(api_key)
            with st.spinner("Buddy is thinking..."):
                response = llm.generate(prompt)
            
            if "Error" in response:
                st.error(response)
            else:
                st.session_state.messages.append({"role": "assistant", "content": response})
                db.add_chat(user_id, "assistant", response)
                
                with st.chat_message("assistant"):
                    st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        # Rerun to show new messages and trigger scroll
        st.rerun()
    
    # Clear button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()