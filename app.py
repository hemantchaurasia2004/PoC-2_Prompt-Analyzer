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
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
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
        instruction_set: str,
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
        
        Task: Analyze the provided system prompt, instruction set, conversation history, and user expectations to suggest improvements.
        
        IMPORTANT: Your response MUST be valid, parseable JSON. Do not include any text before or after the JSON object.
        
        Current System Prompt:
        {system_prompt}
        
        Instruction Set:
        {instruction_set}
        
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
            "rewritten_instructions": "updated instruction set if needed",
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
                analysis = self.validate_and_parse_json(response.content[0].text)
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
                analysis = self.validate_and_parse_json(response.choices[0].message.content)
            
            return analysis
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None

def save_to_history(analysis_data, system_prompt, instruction_set, conversation_history, latest_user_message, agent_response, user_expectation, provider, model):
    """Save the analysis to history with proper management."""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    history_item = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "instruction_set": instruction_set,
        "conversation_history": conversation_history,
        "latest_user_message": latest_user_message,
        "agent_response": agent_response,
        "user_expectation": user_expectation,
        "analysis": analysis_data
    }
    
    # Add to the beginning and maintain only last 10 items
    st.session_state.analysis_history.insert(0, history_item)
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[:10]

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
    
    # Rewritten Prompt and Instructions
    st.subheader("Rewritten System Prompt")
    st.text_area(
        "Copy this improved prompt",
        value=analysis["rewritten_prompt"],
        height=300
    )
    
    if "rewritten_instructions" in analysis:
        st.subheader("Updated Instruction Set")
        st.text_area(
            "Copy these updated instructions",
            value=analysis["rewritten_instructions"],
            height=150
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
    
    # Initialize session state
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
        
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
                with st.expander(f"{hist_item['timestamp']} - {hist_item['provider']}: {rewriter.model_providers[hist_item['provider']][hist_item['model']]}"):
                    st.subheader("Input")
                    st.markdown("**System Prompt:**")
                    st.text(hist_item["system_prompt"])
                    
                    st.markdown("**Instruction Set:**")
                    st.text(hist_item.get("instruction_set", "N/A"))
                    
                    st.markdown("**Latest User Message:**")
                    st.text(hist_item["latest_user_message"])
                    
                    st.markdown("**Agent Response:**")
                    st.text(hist_item["agent_response"])
                    
                    st.markdown("**User Expectation:**")
                    st.text(hist_item["user_expectation"])
                    
                    if st.button("View Full Analysis", key=f"view_{i}"):
                        st.session_state.selected_analysis = hist_item["analysis"]
                        st.rerun()
                    
                    if st.button("Load Into Form", key=f"load_{i}"):
                        st.session_state.load_history_item = hist_item
                        st.rerun()

    # Main content area
    col1, col2 = st.columns(2)
    
    # Check if we need to load a history item
    if "load_history_item" in st.session_state:
        system_prompt = st.session_state.load_history_item["system_prompt"]
        instruction_set = st.session_state.load_history_item.get("instruction_set", "")
        latest_user_message = st.session_state.load_history_item["latest_user_message"]
        agent_response = st.session_state.load_history_item["agent_response"]
        user_expectation = st.session_state.load_history_item["user_expectation"]
        conversation_history = st.session_state.load_history_item.get("conversation_history", [])
        
        del st.session_state.load_history_item
    else:
        system_prompt = ""
        instruction_set = ""
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

        st.subheader("Instruction Set")
        instruction_set = st.text_area(
            "Enter any additional instructions",
            height=100,
            value=instruction_set,
            help="These are additional instructions that guide the model's behavior"
        )

    with col2:
        st.subheader("Conversation History")
        num_exchanges = st.number_input("Number of conversation exchanges", min_value=1, value=1)
        
        conversation_history_display = []
        for i in range(num_exchanges):
            st.markdown(f"**Exchange {i+1}**")
            
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
                instruction_set,
                conversation_history_display,
                latest_user_message,
                agent_response,
                user_expectation,
                provider,
                model
            )

        if analysis:
            save_to_history(
                analysis,
                system_prompt,
                instruction_set,
                conversation_history_display,
                latest_user_message,
                agent_response,
                user_expectation,
                provider,
                model
            )
            
            display_analysis(analysis)
        else:
            st.error("Analysis failed. Please check the error messages above and try again.")
    
    if "selected_analysis" in st.session_state:
        display_analysis(st.session_state.selected_analysis)
        del st.session_state.selected_analysis

if __name__ == "__main__":
    main()