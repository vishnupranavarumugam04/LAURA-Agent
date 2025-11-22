"""Study Mate Page"""
import streamlit as st
import os
import PyPDF2
import json
from utils.llm import SimpleLLM


def extract_pdf_text(pdf_file) -> str:
    """Extract text from PDF"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error: {str(e)}"


def parse_quiz_response(response_text: str):
    """Parse quiz response into structured format"""
    try:
        # Try to extract JSON from response
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        # Parse JSON
        quiz_data = json.loads(response_text.strip())
        
        # Ensure it's a list
        if isinstance(quiz_data, dict):
            quiz_data = [quiz_data]
        
        return quiz_data
    except:
        return None


def render_interactive_quiz(quiz_data):
    """Render interactive quiz with clickable options"""
    
    # Initialize session state for quiz
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = {}
    
    st.markdown("### 🎯 Interactive Quiz")
    st.caption("Click on the option you think is correct")
    
    total_questions = len(quiz_data)
    correct_count = 0
    
    for idx, question_data in enumerate(quiz_data):
        question_num = idx + 1
        question_key = f"q_{idx}"
        
        # Extract question data
        question = question_data.get("question", f"Question {question_num}")
        options = question_data.get("options", [])
        correct_answer = question_data.get("correct_answer", "")
        explanation = question_data.get("explanation", "No explanation provided.")
        
        # Question card
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E1E1E 0%, #2d2d2d 100%); 
                        padding: 20px; border-radius: 15px; margin-bottom: 20px;
                        border-left: 5px solid #667eea;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <div style="color: #667eea; font-weight: 700; font-size: 14px; margin-bottom: 10px;">
                    Question {question_num} of {total_questions}
                </div>
                <div style="color: #FFFFFF; font-size: 16px; font-weight: 600; line-height: 1.6;">
                    {question}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create columns for options (2x2 grid)
        if len(options) >= 4:
            col1, col2 = st.columns(2)
            cols = [col1, col2, col1, col2]
        else:
            cols = [st.columns(1)[0] for _ in range(len(options))]
        
        # Display options as buttons
        for opt_idx, option in enumerate(options):
            with cols[opt_idx % len(cols)]:
                # Check if this question was already answered
                answered = question_key in st.session_state.quiz_submitted
                selected_answer = st.session_state.quiz_answers.get(question_key, "")
                
                # Determine button style based on answer state
                if answered:
                    if option == correct_answer:
                        # Correct answer - green
                        button_type = "primary"
                        emoji = "✅"
                        disabled = True
                    elif option == selected_answer and option != correct_answer:
                        # Wrong answer selected - red
                        button_type = "secondary"
                        emoji = "❌"
                        disabled = True
                    else:
                        # Not selected
                        button_type = "secondary"
                        emoji = "○"
                        disabled = True
                else:
                    button_type = "secondary"
                    emoji = "○"
                    disabled = False
                
                # Create button
                button_label = f"{emoji} {option}"
                if st.button(button_label, key=f"btn_{question_key}_{opt_idx}", 
                           type=button_type, disabled=disabled, use_container_width=True):
                    # Store answer
                    st.session_state.quiz_answers[question_key] = option
                    st.session_state.quiz_submitted[question_key] = True
                    st.rerun()
        
        # Show feedback if answered
        if question_key in st.session_state.quiz_submitted:
            selected = st.session_state.quiz_answers.get(question_key, "")
            
            if selected == correct_answer:
                # Correct answer feedback
                st.success("🎉 **Correct!** Well done!")
                correct_count += 1
            else:
                # Wrong answer feedback
                st.error(f"❌ **Incorrect!** The correct answer is: **{correct_answer}**")
            
            # Show explanation
            st.info(f"💡 **Explanation:** {explanation}")
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Show score if all questions answered
    answered_count = len(st.session_state.quiz_submitted)
    if answered_count == total_questions:
        st.divider()
        
        # Calculate score
        score_percentage = (correct_count / total_questions) * 100
        
        # Score card with color based on performance
        if score_percentage >= 80:
            color = "#00b894"
            grade = "Excellent! 🌟"
        elif score_percentage >= 60:
            color = "#FFA500"
            grade = "Good Job! 👍"
        else:
            color = "#FF4B4B"
            grade = "Keep Practicing! 💪"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}22 0%, {color}44 100%); 
                        padding: 30px; border-radius: 15px; text-align: center;
                        border: 3px solid {color};
                        box-shadow: 0 8px 20px rgba(0,0,0,0.3);">
                <div style="color: {color}; font-size: 48px; font-weight: 900; margin-bottom: 10px;">
                    {correct_count}/{total_questions}
                </div>
                <div style="color: #FFFFFF; font-size: 24px; font-weight: 700; margin-bottom: 5px;">
                    {grade}
                </div>
                <div style="color: #CCCCCC; font-size: 18px;">
                    Score: {score_percentage:.0f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Reset quiz button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Take Quiz Again", type="primary", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = {}
            st.rerun()


def render():
    st.title("📚 Study Mate")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("API Key not configured")
        st.info("Go to Settings to configure your Google Gemini API key")
        return
    
    tab1, tab2 = st.tabs(["PDF Summary", "Generate Quiz"])
    
    with tab1:
        st.subheader("Upload PDF & Get Summary")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file:
            st.write(f"📄 {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
            
            if st.button("Summarize", use_container_width=True, type="primary"):
                with st.spinner("Processing PDF..."):
                    pdf_text = extract_pdf_text(uploaded_file)
                
                if "Error" not in pdf_text:
                    try:
                        llm = SimpleLLM(api_key)
                        with st.spinner("Generating summary..."):
                            summary = llm.summarize(pdf_text, "medium")
                        
                        st.markdown("### Summary")
                        st.write(summary)
                        st.session_state.study_text = pdf_text
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.error(pdf_text)
    
    with tab2:
        st.subheader("Generate Interactive Quiz")
        
        if "study_text" not in st.session_state:
            st.info("Upload a PDF first in the 'PDF Summary' tab")
            return
        
        num_questions = st.slider("Number of Questions", 5, 20, 10)
        
        if st.button("Generate Quiz", use_container_width=True, type="primary"):
            try:
                llm = SimpleLLM(api_key)
                
                # Enhanced prompt for better JSON formatting
                enhanced_prompt = f"""Create EXACTLY {num_questions} multiple choice questions from the provided text.

CRITICAL: Return ONLY a valid JSON array with NO additional text, explanations, or markdown.

Format (return ONLY this, nothing else):
[
  {{
    "question": "What is...",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option B",
    "explanation": "Brief explanation why this is correct"
  }}
]

Rules:
1. Each question must have EXACTLY 4 options
2. correct_answer must be one of the options (exact match)
3. Keep questions clear and concise
4. Explanations should be 1-2 sentences
5. Return ONLY the JSON array, no other text

Text to create questions from:
{st.session_state.study_text[:6000]}"""
                
                with st.spinner("Creating interactive quiz..."):
                    quiz_response = llm.generate(enhanced_prompt, max_tokens=2000)
                
                # Parse the quiz
                quiz_data = parse_quiz_response(quiz_response)
                
                if quiz_data and len(quiz_data) > 0:
                    # Clear previous quiz state
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = {}
                    st.session_state.current_quiz = quiz_data
                    st.success(f"✅ Generated {len(quiz_data)} questions!")
                    st.rerun()
                else:
                    st.error("Failed to generate quiz. Please try again.")
                    with st.expander("Debug: Raw Response"):
                        st.code(quiz_response)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Display quiz if it exists
        if "current_quiz" in st.session_state and st.session_state.current_quiz:
            st.divider()
            render_interactive_quiz(st.session_state.current_quiz)

