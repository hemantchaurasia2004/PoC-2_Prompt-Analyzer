import streamlit as st
import anthropic
from openai import OpenAI
import json
import re
from typing import Dict, List, Optional, Tuple
import datetime

class PromptRewriter:
    def __init__(self):
        """Initialize the prompt rewriter with API clients."""
        self.anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        self.model_providers = {
            "Anthropic": {
                "claude-3-opus-20240229": "Claude 3 Opus",
                "claude-3-sonnet-20240229": "Claude 3 Sonnet",
                "claude-3-haiku-20240307": "Claude 3 Haiku",
                "claude-3-5-haiku-latest": "Claude 3.5 Haiku",
                "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet"
            },
            "OpenAI": {
                "gpt-4-0125-preview": "GPT-4 Turbo",
                "gpt-4": "GPT-4",
                "gpt-3.5-turbo": "GPT-3.5 Turbo",
                "gpt-4o": "GPT-4o"
            }
        }

    def validate_and_parse_json(self, json_str):
        """Validate and parse JSON string, handling common formatting issues."""
        # Remove any potential markdown code block indicators
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find JSON content between curly braces
            json_match = re.search(r'(\{.*\})', json_str, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass
            raise

    def analyze_prompt(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        latest_user_message: str,
        agent_response: str,
        user_expectation: str,
        provider: str,
        model: str
    ) -> Optional[Dict]:
        """
        Analyze the current prompt and suggest improvements based on conversation and expectations.
        """
        analysis_prompt = f"""
        You are an expert in AI system prompt engineering and conversation analysis.
        
        Task: Analyze the provided system prompt, conversation history, and user expectations to suggest improvements.
        
        IMPORTANT: Your response MUST be valid, parseable JSON. Do not include any text before or after the JSON object.
        
        Current System Prompt:
        {system_prompt}
        
        Conversation History:
        {json.dumps(conversation_history, indent=2)}
        
        Latest User Message:
        {latest_user_message}
        
        Agent's Response:
        {agent_response}
        
        User's Expected Behavior:
        {user_expectation}
        
        Provide a detailed analysis and suggestions in the following JSON format:
        {{
            "current_behavior_analysis": {{
                "strengths": [list of current prompt strengths],
                "weaknesses": [list of areas where the prompt failed to meet expectations],
                "gap_analysis": "detailed explanation of the gap between expected and actual behavior"
            }},
            "improvement_suggestions": {{
                "modifications": [
                    {{
                        "section": "affected section of the prompt",
                        "current_text": "current text",
                        "suggested_text": "suggested modification",
                        "reasoning": "explanation for the change"
                    }}
                ],
                "additions": [
                    {{
                        "section": "new section name",
                        "text": "suggested text to add",
                        "reasoning": "explanation for the addition"
                    }}
                ],
                "removals": [
                    {{
                        "section": "section to remove",
                        "text": "text to remove",
                        "reasoning": "explanation for removal"
                    }}
                ]
            }},
            "rewritten_prompt": "complete rewritten system prompt",
            "expected_impact": {{
                "conversation_flow": "how the changes will improve conversation flow",
                "user_satisfaction": "how the changes will better meet user expectations",
                "agent_behavior": "specific behavioral changes expected"
            }}
        }}
        
        Remember, your entire response must be ONLY valid JSON that can be parsed by Python's json.loads() function.
        """

        try:
            if provider == "Anthropic":
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                try:
                    analysis = self.validate_and_parse_json(response.content[0].text)
                except json.JSONDecodeError as json_err:
                    st.error(f"Invalid JSON response from Anthropic model. Error: {json_err}")
                    st.error(f"First 200 characters of response: {response.content[0].text[:200]}...")
                    return None
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in analyzing and improving AI system prompts. Your response must be valid, parseable JSON only."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.2
                )
                try:
                    analysis = self.validate_and_parse_json(response.choices[0].message.content)
                except json.JSONDecodeError as json_err:
                    st.error(f"Invalid JSON response from OpenAI model. Error: {json_err}")
                    st.error(f"First 200 characters of response: {response.choices[0].message.content[:200]}...")
                    return None
            
            return analysis
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None

def format_conversation_history(conversations: List[Dict[str, str]]) -> str:
    """Format conversation history for display."""
    formatted = ""
    for msg in conversations:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted += f"{role.capitalize()}: {content}\n\n"
    return formatted

def save_to_history(analysis_data, system_prompt, conversation_history, latest_user_message, agent_response, user_expectation, provider, model):
    """Save the analysis to history."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    history_item = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "conversation_history": conversation_history,
        "latest_user_message": latest_user_message,
        "agent_response": agent_response,
        "user_expectation": user_expectation,
        "analysis": analysis_data
    }
    
    st.session_state.analysis_history.insert(0, history_item)  # Add to the beginning
    
    # Limit history to 10 items to avoid bloating the session state
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[:10]

def get_example_analyses():
    """Return example analyses for first-time users."""
    return [
        {
            "timestamp": "Example 1",
            "provider": "Anthropic",
            "model": "claude-3-opus-20240229",
            "system_prompt": "You are a helpful customer service assistant for a travel company. Answer user queries about bookings and travel plans.",
            "conversation_history": [],
            "latest_user_message": "x@test.com",
            "agent_response": "Thank you for providing your email address. Could you please let me know your travel dates, including the departure and return dates?",
            "user_expectation": "Actually - it shouldn't accept such email address. Instead it should validate and not accept emails which are spam or inaccurate.",
            "analysis": {
                "current_behavior_analysis": {
                    "strengths": ["Polite and professional tone", "Clear request for additional information"],
                    "weaknesses": ["Failed to validate email format", "Accepted obviously invalid email", "No security or validation protocols"],
                    "gap_analysis": "The assistant accepted an obviously invalid email format (x@test.com) without any verification or validation. This creates security risks and could lead to issues with booking confirmations later. The system should validate email inputs before proceeding with the conversation."
                },
                "improvement_suggestions": {
                    "modifications": [
                        {
                            "section": "Email validation",
                            "current_text": "You are a helpful customer service assistant for a travel company. Answer user queries about bookings and travel plans.",
                            "suggested_text": "You are a helpful customer service assistant for a travel company. Answer user queries about bookings and travel plans. Always validate user-provided information, particularly email addresses, before proceeding with the conversation. Reject obviously invalid email formats and explain why they cannot be accepted.",
                            "reasoning": "Adding explicit instructions about validation will ensure the assistant checks the validity of critical information like email addresses."
                        }
                    ],
                    "additions": [
                        {
                            "section": "Email format validation",
                            "text": "When a user provides an email address, verify that it follows proper email format (username@domain.com). The email should have a username, @ symbol, and a proper domain with extension. Reject suspicious or obviously invalid emails such as test@test.com, x@y.com, or emails with no domain extension.",
                            "reasoning": "This addition provides specific guidelines on what constitutes a valid email format and examples of invalid formats to reject."
                        }
                    ],
                    "removals": []
                },
                "rewritten_prompt": "You are a helpful customer service assistant for a travel company. Answer user queries about bookings and travel plans. Always validate user-provided information, particularly email addresses, before proceeding with the conversation. Reject obviously invalid email formats and explain why they cannot be accepted.\n\nWhen a user provides an email address, verify that it follows proper email format (username@domain.com). The email should have a username, @ symbol, and a proper domain with extension. Reject suspicious or obviously invalid emails such as test@test.com, x@y.com, or emails with no domain extension.\n\nIf a user provides invalid information, politely ask them to provide correct information and explain why the validation is important for successful booking and communication.",
                "expected_impact": {
                    "conversation_flow": "The conversation will include validation steps that ensure accurate information collection before proceeding with travel planning.",
                    "user_satisfaction": "Users will appreciate the security measures and will have greater confidence that their booking details are being handled professionally.",
                    "agent_behavior": "The agent will now identify and reject invalid email formats, request proper information, and explain the importance of accurate contact details."
                }
            }
        },
        {
            "timestamp": "Example 2",
            "provider": "OpenAI",
            "model": "gpt-4o",
            "system_prompt": "You are a math tutor helping students with algebra problems.",
            "conversation_history": [
                {"role": "user", "content": "I need help with factoring: x² - 9"},
                {"role": "assistant", "content": "This is a perfect square trinomial. The answer is (x+3)(x-3)."}
            ],
            "latest_user_message": "Why?",
            "agent_response": "It's the difference of squares formula: a² - b² = (a+b)(a-b). Here, a = x and b = 3, so x² - 9 = x² - 3² = (x+3)(x-3).",
            "user_expectation": "The assistant should have explained the steps more clearly and checked if I understood before giving the answer so quickly.",
            "analysis": {
                "current_behavior_analysis": {
                    "strengths": ["Provided correct mathematical explanation", "Used appropriate algebraic formula"],
                    "weaknesses": ["Too direct and brief", "No step-by-step breakdown", "Didn't check for student understanding"],
                    "gap_analysis": "The assistant provided a mathematically correct but pedagogically ineffective response. As a math tutor, the primary goal should be teaching the student to understand the concept, not just providing the answer. The assistant failed to break down the steps, check for understanding, or engage the student in the learning process."
                },
                "improvement_suggestions": {
                    "modifications": [
                        {
                            "section": "Teaching approach",
                            "current_text": "You are a math tutor helping students with algebra problems.",
                            "suggested_text": "You are a patient and thorough math tutor helping students understand algebra concepts. Focus on teaching the process and building understanding rather than just providing answers. Break down problems step-by-step and check for student comprehension before moving forward.",
                            "reasoning": "This modification emphasizes the pedagogical role of the tutor and the importance of process over answers."
                        }
                    ],
                    "additions": [
                        {
                            "section": "Response structure",
                            "text": "When responding to math questions: 1) Acknowledge the question, 2) Explain the underlying concept, 3) Work through the solution step-by-step, 4) Summarize the approach, and 5) Check for understanding by asking if any part needs further clarification.",
                            "reasoning": "This addition provides a clear structure for responses that ensures thorough explanations and student engagement."
                        }
                    ],
                    "removals": []
                },
                "rewritten_prompt": "You are a patient and thorough math tutor helping students understand algebra concepts. Focus on teaching the process and building understanding rather than just providing answers. Break down problems step-by-step and check for student comprehension before moving forward.\n\nWhen responding to math questions: 1) Acknowledge the question, 2) Explain the underlying concept, 3) Work through the solution step-by-step, 4) Summarize the approach, and 5) Check for understanding by asking if any part needs further clarification.\n\nUse visual explanations when helpful (like showing the distribution of terms) and relate concepts to previous material when appropriate. Encourage the student to think through parts of the problem themselves with guided questions.",
                "expected_impact": {
                    "conversation_flow": "Conversations will become more interactive, with the tutor guiding the student through the learning process rather than simply providing answers.",
                    "user_satisfaction": "Students will gain deeper understanding and feel more supported in their learning journey.",
                    "agent_behavior": "The agent will provide more comprehensive explanations, break down problems into manageable steps, and actively check for student understanding."
                }
            }
        }
    ]

def display_analysis(analysis):
    """Display the analysis results."""
    if not analysis:
        return
        
    st.header("Analysis Results")
    
    # Current Behavior Analysis
    st.subheader("Current Behavior Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths**")
        for strength in analysis["current_behavior_analysis"]["strengths"]:
            st.markdown(f"- {strength}")
    with col2:
        st.markdown("**Weaknesses**")
        for weakness in analysis["current_behavior_analysis"]["weaknesses"]:
            st.markdown(f"- {weakness}")
    
    st.markdown("**Gap Analysis**")
    st.write(analysis["current_behavior_analysis"]["gap_analysis"])
    
    # Improvement Suggestions
    st.subheader("Suggested Improvements")
    
    if analysis["improvement_suggestions"]["modifications"]:
        st.markdown("**Modifications**")
        for mod in analysis["improvement_suggestions"]["modifications"]:
            with st.expander(f"Modify: {mod['section']}"):
                st.markdown("**Current Text:**")
                st.text(mod["current_text"])
                st.markdown("**Suggested Text:**")
                st.text(mod["suggested_text"])
                st.markdown("**Reasoning:**")
                st.write(mod["reasoning"])
    
    if analysis["improvement_suggestions"]["additions"]:
        st.markdown("**Additions**")
        for add in analysis["improvement_suggestions"]["additions"]:
            with st.expander(f"Add: {add['section']}"):
                st.markdown("**Suggested Text:**")
                st.text(add["text"])
                st.markdown("**Reasoning:**")
                st.write(add["reasoning"])
    
    if analysis["improvement_suggestions"]["removals"]:
        st.markdown("**Removals**")
        for rem in analysis["improvement_suggestions"]["removals"]:
            with st.expander(f"Remove: {rem['section']}"):
                st.markdown("**Text to Remove:**")
                st.text(rem["text"])
                st.markdown("**Reasoning:**")
                st.write(rem["reasoning"])
    
    # Rewritten Prompt
    st.subheader("Rewritten System Prompt")
    st.text_area(
        "Copy this improved prompt",
        value=analysis["rewritten_prompt"],
        height=300
    )
    
    # Expected Impact
    st.subheader("Expected Impact")
    impact = analysis["expected_impact"]
    st.markdown("**Conversation Flow:**")
    st.write(impact["conversation_flow"])
    st.markdown("**User Satisfaction:**")
    st.write(impact["user_satisfaction"])
    st.markdown("**Agent Behavior:**")
    st.write(impact["agent_behavior"])

def main():
    st.title("AI System Prompt Rewriter")
    
    # Initialize session state for history
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
        
    # Initialize first-time user examples
    if "examples_loaded" not in st.session_state:
        st.session_state.examples_loaded = True
        st.session_state.analysis_history = get_example_analyses()
    
    rewriter = PromptRewriter()

    # Create tabs for the sidebar
    tab1, tab2 = st.sidebar.tabs(["Model Selection", "History"])
    
    with tab1:
        st.header("Model Selection")
        provider = st.selectbox(
            "Select Provider",
            options=list(rewriter.model_providers.keys())
        )
        model = st.selectbox(
            "Select Model",
            options=list(rewriter.model_providers[provider].keys()),
            format_func=lambda x: rewriter.model_providers[provider][x]
        )
    
    with tab2:
        st.header("Analysis History")
        if not st.session_state.analysis_history:
            st.info("No analysis history yet. Run an analysis to see it here.")
        else:
            for i, hist_item in enumerate(st.session_state.analysis_history):
                with st.expander(f"{hist_item['timestamp']} - {hist_item['provider']}: {rewriter.model_providers[hist_item['provider']][hist_item['model']] if hist_item['provider'] in ['Anthropic', 'OpenAI'] else hist_item['model']}"):
                    st.subheader("Input")
                    st.markdown("**System Prompt:**")
                    st.text(hist_item["system_prompt"])
                    
                    st.markdown("**Latest User Message:**")
                    st.text(hist_item["latest_user_message"])
                    
                    st.markdown("**Agent Response:**")
                    st.text(hist_item["agent_response"])
                    
                    st.markdown("**User Expectation:**")
                    st.text(hist_item["user_expectation"])
                    
                    # Add a button to load this analysis for viewing
                    if st.button("View Full Analysis", key=f"view_{i}"):
                        st.session_state.selected_analysis = hist_item["analysis"]
                        st.rerun()
                    
                    # Add a button to load this history item into the input fields
                    if st.button("Load Into Form", key=f"load_{i}"):
                        st.session_state.load_history_item = hist_item
                        st.rerun()

    # Main content area
    col1, col2 = st.columns(2)
    
    # Check if we need to load a history item
    if "load_history_item" in st.session_state:
        system_prompt = st.session_state.load_history_item["system_prompt"]
        latest_user_message = st.session_state.load_history_item["latest_user_message"]
        agent_response = st.session_state.load_history_item["agent_response"]
        user_expectation = st.session_state.load_history_item["user_expectation"]
        conversation_history = st.session_state.load_history_item.get("conversation_history", [])
        
        # Clear the flag after loading
        del st.session_state.load_history_item
    else:
        system_prompt = ""
        latest_user_message = ""
        agent_response = ""
        user_expectation = ""
        conversation_history = []
    
    with col1:
        st.subheader("Current System Prompt")
        system_prompt = st.text_area(
            "Enter the current system prompt",
            height=200,
            value=system_prompt
        )

    with col2:
        st.subheader("Conversation History")
        num_exchanges = st.number_input("Number of conversation exchanges", min_value=1, value=1)
        
        conversation_history_display = []
        for i in range(num_exchanges):
            st.markdown(f"**Exchange {i+1}**")
            
            # Pre-fill from history if available
            user_val = conversation_history[i*2]["content"] if conversation_history and i*2 < len(conversation_history) else ""
            agent_val = conversation_history[i*2+1]["content"] if conversation_history and i*2+1 < len(conversation_history) else ""
            
            user_msg = st.text_area(f"User message {i+1}", key=f"user_{i}", value=user_val)
            agent_msg = st.text_area(f"Agent response {i+1}", key=f"agent_{i}", value=agent_val)
            
            if user_msg:
                conversation_history_display.append({"role": "user", "content": user_msg})
            if agent_msg:
                conversation_history_display.append({"role": "assistant", "content": agent_msg})

    st.subheader("Latest Interaction")
    latest_user_message = st.text_area("Latest user message", value=latest_user_message)
    agent_response = st.text_area("Agent's response", value=agent_response)
    
    st.subheader("User Expectations")
    user_expectation = st.text_area(
        "Describe how you expected the agent to respond and why",
        value=user_expectation,
        placeholder="e.g., The agent should have moved to step 3 because..."
    )

    if st.button("Analyze and Suggest Improvements"):
        if not all([system_prompt, latest_user_message, agent_response, user_expectation]):
            st.warning("Please fill in all required fields.")
            return

        with st.spinner("Analyzing prompt and generating suggestions..."):
            analysis = rewriter.analyze_prompt(
                system_prompt,
                conversation_history_display,
                latest_user_message,
                agent_response,
                user_expectation,
                provider,
                model
            )

        if analysis:
            # Save to history
            save_to_history(
                analysis,
                system_prompt,
                conversation_history_display,
                latest_user_message,
                agent_response,
                user_expectation,
                provider,
                model
            )
            
            # Display analysis results
            display_analysis(analysis)
        else:
            st.error("Analysis failed. Please check the error messages above and try again.")
    
    # Display selected analysis from history if available
    if "selected_analysis" in st.session_state:
        display_analysis(st.session_state.selected_analysis)
        # Clear the selection after displaying
        del st.session_state.selected_analysis

if __name__ == "__main__":
    main()