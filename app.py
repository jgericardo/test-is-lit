import json
import time
from datetime import datetime

import streamlit as st
import openai
from annotated_text import annotated_text

# Set page config
st.set_page_config(
    page_title="Harmful Speech Detector",
    page_icon="🛡️",
    layout="wide"
)

# App title and description
st.title("🛡️ Harmful Speech Detector")
st.markdown("Enter text to check for harmful content. The app will analyze your text and suggest improvements.")

# Create a text area for user input
user_text = st.text_area("Enter your text here:", height=250)

# Create a placeholder for results
result_container = st.container()

# About section in sidebar
with st.sidebar:
    st.subheader("About")
    st.write("This app uses a custom LLM to detect harmful speech and suggest improvements.")
    st.write("Types of harmful content detected:")
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
                    "tagged_text": response_content,
                    "harmful_elements": [],
                    "reworded_sentence": ""
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
def call_api(text):
    """Call the API to analyze text using OpenAI client"""
    # Validate input before processing
    if not text or text.isspace():
        return {
            "harmful": False,
            "tagged_text": "",
            "harmful_elements": [],
            "reworded_sentence": ""
        }
        
    try:
        # API configuration
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "your_api_key_here")
        OPENAI_BASE_URL = st.secrets.get("OPENAI_BASE_URL", "your_base_url_here")
        
        CHAT_PROMPT_TEMPLATE = st.secrets.get("CHAT_PROMPT_TEMPLATE", "your_chat_prompt_template_here")
        MODEL_NAME = st.secrets.get("MODEL_NAME", "your_model_name_here")
        TEMPERATURE = st.secrets.get("TEMPERATURE", "your_temperature_here")
        
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY, 
            base_url=OPENAI_BASE_URL,
        )

        final_message = CHAT_PROMPT_TEMPLATE + text
        
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": final_message}
            ],
            temperature=TEMPERATURE,
        )
        elapsed_time = time.perf_counter() - start_time
        print(f"Execution time: {elapsed_time:.6f} seconds")

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
                "tagged_text": text,
                "harmful_elements": []
            }
            
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return {
            "harmful": False,
            "tagged_text": text,
            "harmful_elements": [],
            "reworded_sentence": ""
        }

# Function to generate a revised text based on the harmful elements
def generate_revised_text(original_text, harmful_elements):
    """Replace harmful elements with their alternatives"""
    revised_text = original_text
    
    # Sort elements by their position in reverse order to avoid index shifting
    # We need to find the positions first
    text_lower = original_text.lower()
    elements_with_pos = []
    
    for element in harmful_elements:
        term = element["text"].lower()
        start_pos = text_lower.find(term)
        if start_pos != -1:
            elements_with_pos.append({
                "text": element["text"],
                "start": start_pos,
                "end": start_pos + len(term),
                "alternative": element["alternative"]
            })
    
    # Sort by start position in reverse order
    elements_with_pos.sort(key=lambda x: x["start"], reverse=True)
    
    # Replace each harmful element with its alternative
    for element in elements_with_pos:
        revised_text = (
            revised_text[:element["start"]] + 
            element["alternative"] + 
            revised_text[element["end"]:]
        )
    
    return revised_text

# Button to analyze text
if st.button("Analyze Text 🔎"):
    # Check if the input is blank or contains only whitespace
    if not user_text or user_text.isspace():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Get analysis results from API
            results = call_api(user_text)
            
            with result_container:
                st.subheader("Analysis Results")
                
                # Show safety status
                if results["harmful"]:
                    st.error("⚠️ Potentially harmful content detected")
                else:
                    st.success("✅ Content appears safe")
            
            # Display annotated text if harmful content is found
            if results["harmful"]:
                st.markdown("### Highlighted Issues:")
                
                # Convert harmful_elements to a format suitable for annotation
                harmful_elements = results["harmful_elements"]
                
                # We need to find the positions of each harmful element in the text
                text_lower = user_text.lower()
                segments = []
                
                for element in harmful_elements:
                    term = element["text"].lower()
                    start_pos = 0
                    
                    while start_pos < len(text_lower):
                        pos = text_lower.find(term, start_pos)
                        if pos == -1:
                            break
                        
                        segments.append({
                            "text": user_text[pos:pos+len(term)],
                            "start": pos,
                            "end": pos + len(term),
                            "type": element["type"],
                            "alternative": element["alternative"]
                        })
                        
                        start_pos = pos + len(term)
                
                # Sort segments by starting position
                segments.sort(key=lambda x: x["start"])
                
                # Build annotated text components
                annotated_components = []
                last_end = 0
                
                for segment in segments:
                    # Add text before the harmful segment
                    if segment["start"] > last_end:
                        annotated_components.append(user_text[last_end:segment["start"]])
                    
                    # Add the harmful segment with annotation
                    harmful_text = segment["text"]
                    
                    # Set color based on type
                    color = {
                        "threat": "#F08080",           # Light coral
                        "insult": "#FF9AA2",           # Soft red
                        "severe_toxicity": "#DC143C",  # Crimson
                        "toxicity": "#FFB7B2",         # Salmon
                        "profanity": "#FFDAC1",        # Peach
                        "sexually_explicit": "#DB7093", # Pale violet red
                        "identity_attack": "#DDA0DD",  # Plum
                        "flirtation": "#D8BFD8",       # Thistle
                        "passive_aggressive": "#FDFD96", # Yellow
                        "sarcasm": "#20B2AA", # Light Sea Green
                        "harmful_content": "#faa"      # Default
                    }.get(segment["type"], "#faa")
                    
                    annotated_components.append((harmful_text, segment["type"], color))
                    
                    last_end = segment["end"]
                
                # Add any remaining text
                if last_end < len(user_text):
                    annotated_components.append(user_text[last_end:])
                
                # Display the annotated text
                annotated_text(*annotated_components)
                
                # Generate and show suggested revision
                suggested_text = generate_revised_text(user_text, results["harmful_elements"])
                
                # Show alternatives for each harmful element
                st.markdown("### Alternative Suggestions:")
                for element in results["harmful_elements"]:
                    st.markdown(f"- Replace '**{element['text']}**' with '**{element['alternative']}**' ({element['type']})")
                
                st.markdown("### Suggested Revision:")
                st.success(results["reworded_message"])
            
            # Show raw analysis (can be hidden in production)
            with st.expander("Show detailed analysis"):
                st.json(results)
                st.text(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")