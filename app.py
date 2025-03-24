import json
import time
from datetime import datetime

import streamlit as st
import openai
from annotated_text import annotated_text

# Set page config
st.set_page_config(
    page_title="Writing Assistant",
    page_icon="🛡️",
    layout="wide"
)

# App title and description
st.title("💼 Writing Assistant")
st.markdown("Enter your text to check for appropriateness and get suggestions for professional communications.")

# Create parameters for the user to configure
col1, col2, col3 = st.columns(3)
with col1:
    tone = st.selectbox("Tone", options=["Formal", "Professional", "Casual", "Friendly"], index=0)
with col2:
    audience = st.selectbox("Audience", options=["Clients", "Colleagues", "Management", "Team", "General"], index=0)
with col3:
    length = st.selectbox("Length", options=["Concise", "Standard", "Detailed"], index=1)

# Create a text area for user input
user_text = st.text_area("Enter your text here:", height=250)

# Create a placeholder for results
result_container = st.container()

# About section in sidebar
with st.sidebar:
    st.subheader("About")
    st.write("This app helps improve your business communications by analyzing text for problematic content and suggesting improvements.")
    st.write("Types of problematic content detected:")
    st.markdown("- Threats")
    st.markdown("- Insults") 
    st.markdown("- Toxicity")
    st.markdown("- Profanity")
    st.markdown("- Sexually explicit content")
    st.markdown("- Identity attacks")
    st.markdown("- Flirtation")
    st.markdown("- Passive Aggressive")
    st.markdown("- Sarcasm")
    
    st.divider()
    st.write("Developed by **The Attic AI** as a proof of concept.")

    # Add a placeholder for analysis time
    analysis_time_placeholder = st.empty()
    
    # Add a placeholder for analysis time only
    analysis_time_placeholder = st.empty()

def sanitize_json_response(response_content):
    """
    Sanitize JSON response to handle common issues including unescaped quotes
    and trailing commas.
    
    Args:
        response_content (str): The raw response content from the API
        
    Returns:
        dict: The parsed JSON as a Python dictionary
    """
    # Clean up the response content
    cleaned_content = response_content.replace("```python", "").replace("```json", "").replace("```", "").strip()
    cleaned_content = cleaned_content.replace("True", "true").replace("False", "false")
    
    # Fix trailing commas - common issue with JSON from language models
    import re
    cleaned_content = re.sub(r',(\s*[\}\]])', r'\1', cleaned_content)
    
    # Try to parse the JSON directly first
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        # If direct parsing fails, try the ast.literal_eval approach
        # which can handle more Python-like dictionary syntax
        try:
            import ast
            result = ast.literal_eval(cleaned_content)
            return result
        except (SyntaxError, ValueError):
            # If that fails too, try to fix common quote issues more aggressively
            try:
                # This is a last-resort approach that tries to fix unbalanced quotes
                # It's not perfect but can handle many common cases
                fixed_content = fix_unbalanced_quotes(cleaned_content)
                return json.loads(fixed_content)
            except Exception:
                # If all else fails, return a default structure
                return {
                    "harmful": False,
                    "harmful_sentences": [],
                    "reworded_message": ""
                }

def fix_unbalanced_quotes(json_str):
    """
    Attempt to fix JSON with unbalanced or unescaped quotes.
    This is a best-effort function for last-resort cases.
    
    Args:
        json_str (str): The JSON string to fix
        
    Returns:
        str: A potentially fixed JSON string
    """
    import re
    
    # State tracking
    in_string = False
    escape_next = False
    result = []
    
    # Process character by character
    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            result.append(char)
            escape_next = True
            continue
            
        if char == '"' and not in_string:
            in_string = True
            result.append(char)
            continue
            
        if char == '"' and in_string:
            in_string = False
            result.append(char)
            continue
            
        # If we're in a string and encounter an unescaped quote, escape it
        if in_string and char in ['"', "'"]:
            result.append('\\')
        
        result.append(char)
    
    # If we end while still in a string, close it
    if in_string:
        result.append('"')
    
    return ''.join(result)

# Function to call the API using OpenAI client
def call_api(text, tone, audience, length):
    """Call the API to analyze text using OpenAI client"""
    # Validate input before processing
    if not text or text.isspace():
        return {
            "harmful": False,
            "harmful_sentences": [],
            "reworded_message": ""
        }
        
    try:
        # API configuration
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "your_api_key_here")
        RUNPOD_BASE_URL = st.secrets.get("RUNPOD_BASE_URL", "your_base_url_here")
        
        # The prompt template to analyze harmful content
        chat_prompt_template = """
        You're an AI writing assistant specializing in improving business communications. Your task is to thoroughly analyze each sentence for problematic content, while ensuring the final reworded message maintains natural flow and context.
        
        Parameters:
        - Tone: {tone}
        - Context: {audience} 
        - Alternative Length: {length}
        
        For the input text:
        1. First pass - Identify problematic sentences by analyzing EACH SENTENCE individually
        2. Flag ANY sentence containing unprofessional language, personal attacks, threats, or inappropriate content
        3. Be especially alert for:
           - Sentences that attack individuals or teams by name
           - Statements threatening job security or creating a hostile environment
           - Unprofessional or demeaning language, even if subtle
           - Passive-aggressive phrasings that undermine team cohesion
        4. Second pass - Consider the text as a whole to ensure coherent flow
           - Ensure sentence transitions make sense in the reworded message
           - Maintain the original meaning and purpose where appropriate
           - Preserve the logical structure of paragraphs and arguments
           - Create a cohesive final message that reads naturally
        
        Output this exact JSON format:
        {{
            "harmful": true/false,
            "harmful_sentences": [
                {{
                    "sentence": "problematic sentence",
                    "type": "threat/insult/severe_toxicity/toxicity/profanity/sexually_explicit/identity_attack/flirtation/passive_aggressive/sarcasm",
                    "severity": "Critical/Non-critical", 
                    "explanation": "Brief explanation of why the statement or content is problematic",
                    "recommendation": "fix/remove", 
                    "alternative": "suggested replacement that flows with surrounding context (LEAVE blank if the recommendation is remove)"
                }}
            ],
            "reworded_message": "Improved version of entire text with natural transitions and coherent flow"
        }}
        
        # If parameters are missing, use these defaults:
        # Tone: formal
        # Context: general
        # Length: standard
        
        Example Input (Team Communication):
        Tone: Professional
        Context: Team
        Length: Standard
        
        "The marketing team's latest campaign was a complete disaster. Whoever designed those graphics should be fired. I'm sick of having to fix everyone's mistakes. The client was furious and we might lose the account because of your incompetence. Let's meet tomorrow to discuss how to clean up this mess."
        
        Example Output:
        {{
            "harmful": true,
            "harmful_sentences": [
                {{
                    "sentence": "The marketing team's latest campaign was a complete disaster.",
                    "type": "toxicity",
                    "severity": "Non-critical",
                    "explanation": "Uses extreme negative language that can damage team morale and doesn't offer constructive feedback",
                    "recommendation": "fix",
                    "alternative": "The marketing team's latest campaign didn't meet our expected outcomes."
                }},
                {{
                    "sentence": "Whoever designed those graphics should be fired.",
                    "type": "threat",
                    "severity": "Critical",
                    "explanation": "Contains a direct threat to job security and publicly criticizes team member(s) in a way that creates a hostile work environment",
                    "recommendation": "remove",
                    "alternative": ""
                }},
                {{
                    "sentence": "I'm sick of having to fix everyone's mistakes.",
                    "type": "insult",
                    "severity": "Critical",
                    "explanation": "Includes a broad, personal attack on the team's competence and expresses inappropriate frustration",
                    "recommendation": "remove",
                    "alternative": ""
                }},
                {{
                    "sentence": "The client was furious and we might lose the account because of your incompetence.",
                    "type": "identity_attack",
                    "severity": "Critical",
                    "explanation": "Directly blames team members and labels them as incompetent, which is demeaning and creates an accusatory environment",
                    "recommendation": "fix",
                    "alternative": "The client expressed strong concerns about the work, which puts the account at risk."
                }},
                {{
                    "sentence": "Let's meet tomorrow to discuss how to clean up this mess.",
                    "type": "passive_aggressive",
                    "severity": "Non-critical",
                    "explanation": "The phrase 'clean up this mess' frames the situation in unnecessarily negative terms",
                    "recommendation": "fix",
                    "alternative": "Let's meet tomorrow to develop an action plan to address these issues."
                }}
            ],
            "reworded_message": "The marketing team's latest campaign didn't meet our expected outcomes. The client expressed strong concerns about the work, which puts the account at risk. Let's meet tomorrow to develop an action plan to address these issues."
        }}
        
        Now analyze this text:
        """
        
        # Replace the parameter placeholders
        formatted_prompt = chat_prompt_template.format(
            tone=tone,
            audience=audience,
            length=length
        )
        
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY, 
            base_url=RUNPOD_BASE_URL,
        )

        final_message = formatted_prompt + text
        
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model="microsoft/phi-4",
            messages=[
                {"role": "user", "content": final_message}
            ],
            temperature=0.0,
        )
        elapsed_time = time.perf_counter() - start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")
        
        # Store the elapsed time in session state to access it later
        st.session_state.last_analysis_time = elapsed_time

        # Parse the response content as JSON
        try:
            response_content = response.choices[0].message.content
            if "```python" in response_content:
                response_content = response_content.replace("True", "true").replace("False", "false")
            response_content = response_content.replace("```python", "").replace("```json", "")
            response_content = response_content.replace("```", "").strip()
            # Use the sanitizer to parse the response
            result = sanitize_json_response(response_content)
            return result
        except json.JSONDecodeError:
            st.error("Failed to parse API response as JSON")
            st.code(response_content)  # Show the raw response for debugging
            return {
                "harmful": False,
                "harmful_sentences": [],
                "reworded_message": ""
            }
            
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return {
            "harmful": False,
            "harmful_sentences": [],
            "reworded_message": ""
        }

# Function to count critical and non-critical issues
def count_issues(harmful_sentences):
    """Count the number of critical and non-critical issues"""
    total_issues = len(harmful_sentences)
    critical_issues = sum(1 for sentence in harmful_sentences if sentence["severity"].lower() == "critical")
    non_critical_issues = total_issues - critical_issues
    
    return total_issues, critical_issues, non_critical_issues

# Function to annotate text with problematic sentences
def create_annotated_text(original_text, harmful_sentences):
    """Create annotated text components highlighting problematic sentences"""
    if not harmful_sentences:
        return [original_text]
    
    # Find each problematic sentence in the original text
    annotated_components = []
    last_end = 0
    
    # Sort sentences by where they appear in the text
    segments = []
    for sentence_data in harmful_sentences:
        sentence = sentence_data["sentence"]
        start_pos = original_text.find(sentence, last_end)
        if start_pos != -1:
            segments.append({
                "text": sentence,
                "start": start_pos,
                "end": start_pos + len(sentence),
                "type": sentence_data["type"],
                "severity": sentence_data["severity"],
                "alternative": sentence_data["alternative"]
            })
    
    # Sort segments by starting position
    segments.sort(key=lambda x: x["start"])
    
    # Build annotated text components
    last_end = 0
    for segment in segments:
        # Add text before the harmful segment
        if segment["start"] > last_end:
            annotated_components.append(original_text[last_end:segment["start"]])
        
        # Add the harmful segment with annotation
        harmful_text = segment["text"]
        
        # Set color based on type and severity
        severity_colors = {
            "Critical": {
                "threat": "#FF6B6B",         # Bright red
                "insult": "#FF9AA2",         # Soft red 
                "severe_toxicity": "#DC143C", # Crimson
                "toxicity": "#FF7F7F",       # Light coral
                "profanity": "#FFCCCB",      # Light red
                "sexually_explicit": "#DB7093", # Pale violet red
                "identity_attack": "#DDA0DD", # Plum
                "flirtation": "#D8BFD8",     # Thistle
                "passive_aggressive": "#FFD700", # Gold
                "sarcasm": "#FFA07A"         # Light salmon
            },
            "Non-critical": {
                "threat": "#FFC3C3",         # Very light red
                "insult": "#FFD1D7",         # Very light pink
                "severe_toxicity": "#FFB6C1", # Light pink
                "toxicity": "#FFE4E1",       # Misty rose
                "profanity": "#FFDAB9",      # Peach puff
                "sexually_explicit": "#E6E6FA", # Lavender
                "identity_attack": "#F0E6FF", # Very light purple
                "flirtation": "#FFF0F5",     # Lavender blush
                "passive_aggressive": "#FFFACD", # Lemon chiffon 
                "sarcasm": "#FAEBD7"         # Antique white
            }
        }
        
        color = severity_colors.get(segment["severity"], {}).get(segment["type"], "#FAA")
        # Display the type and severity (in uppercase) separately
        label = f"{segment['type']} - {segment['severity'].upper()}"
        
        annotated_components.append((harmful_text, label, color))
        
        last_end = segment["end"]
    
    # Add any remaining text
    if last_end < len(original_text):
        annotated_components.append(original_text[last_end:])
    
    return annotated_components

# Initialize session state for analysis time if it doesn't exist
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = None

# Button to analyze text
if st.button("Analyze Text 🔎"):
    # Check if the input is blank or contains only whitespace
    if not user_text or user_text.isspace():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Get analysis results from API
            results = call_api(user_text, tone, audience, length)
            
            # Count the issues
            total_issues, critical_issues, non_critical_issues = count_issues(results["harmful_sentences"])
            
            # Update the analysis time in the sidebar
            if st.session_state.last_analysis_time:
                with st.sidebar:
                    analysis_time_placeholder.info(f"⏱️ Analysis Time: {st.session_state.last_analysis_time:.2f} seconds")
                    
                    # Only display analysis time in sidebar
                    analysis_time_placeholder.info(f"⏱️ Analysis Time: {st.session_state.last_analysis_time:.2f} seconds")
            
            with result_container:
                st.subheader("Analysis Results")
                
                # Show safety status
                if results["harmful"]:
                    st.error("⚠️ Communication issues detected")
                    
                    # Add issue counters in columns with the same style as sidebar
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info(f"📊 Total Issues Found: {total_issues}")
                    
                    with col2:
                        # Use error color for critical issues
                        if critical_issues > 0:
                            st.error(f"🚨 Critical Issues: {critical_issues}")
                        else:
                            st.success("🚨 Critical Issues: 0")
                    
                    with col3:
                        # Use warning color for non-critical issues
                        if non_critical_issues > 0:
                            st.warning(f"⚠️ Non-critical Issues: {non_critical_issues}")
                        else:
                            st.success("⚠️ Non-critical Issues: 0")
                else:
                    st.success("✅ Communication appears appropriate")
            
            # Display annotated text if harmful content is found
            if results["harmful"]:
                # Show detailed breakdown of issues
                st.markdown("### Issue Breakdown:")
                
                for i, sentence in enumerate(results["harmful_sentences"]):
                    with st.expander(f"Issue {i+1}: {sentence['type'].title()}", expanded=True):
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"**Original:** {sentence['sentence']}")
                            st.markdown(f"**Explanation:** {sentence['explanation']}")
                            st.markdown(f"**Suggestion:** {sentence['alternative']}")
                        with cols[1]:
                            # Color-coded severity using annotated_text
                            st.markdown("**Severity:**")
                            if sentence['severity'].lower() == 'critical':
                                annotated_text(("CRITICAL", "", "#FF6B6B"))
                            else:
                                annotated_text(("NON-CRITICAL", "", "#FFFACD"))
                            st.markdown(f"**Recommendation:** {sentence['recommendation'].title()}")
                
                # Show the reworded message
                st.markdown("### Improved Version:")
                st.success(results["reworded_message"])
            
            # Show raw analysis (can be hidden in production)
            with st.expander("Show detailed analysis"):
                st.json(results)
                st.text(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")