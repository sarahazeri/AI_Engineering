import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, APIError
from tools import Canvas, make_diagram_tools

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Diagram Agent UI", layout="wide")
st.title("📊 Diagram Agent - Chatbot & Live Canvas")

# 1. Initialize Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are an expert Diagram Agent. Your task is to turn user prompts into structured diagram elements.
- Always generate unique element IDs (e.g., 'rect_signup', 'arrow_1').
- For 'arrow' elements, both 'from' and 'to' fields are STRICTLY REQUIRED and MUST be valid element IDs.
- For nodes (rectangle, diamond, circle), always provide a string 'label'.
- Use generate_diagram when creating a brand new diagram from scratch.
- Use modify_diagram when updating properties of an existing node.
- Use add_elements when inserting new nodes without redrawing existing ones.
"""

MAX_STEPS = 10
MAX_MESSAGES = 12

# 2. Manage Session State to persist chat history and canvas
if "canvas" not in st.session_state:
    st.session_state.canvas = Canvas("output_canvas.svg")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Message compression function
def compress_messages(messages: list, max_history: int = MAX_MESSAGES) -> list:
    if len(messages) <= max_history:
        return messages
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    recent_messages = messages[-max_history:]
    if system_msg and recent_messages[0].get("role") != "system":
        return [system_msg] + recent_messages
    return recent_messages

# 4. Build the two-column layout (Split Screen)
col_left, col_right = st.columns([1, 1])

# ----------------- Right column: Chatbot -----------------
with col_right:
    st.subheader("💬 Chat with the Agent")

    # Display message history
    for msg in st.session_state.messages:
        # Check data type to get role and content
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

        # Condition for displaying the message
        if role in ["user", "assistant"] and content:
            with st.chat_message(role):
                st.write(content)

    # User input
    user_input = st.chat_input("Describe the diagram you want...")

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Update system prompt with the current Canvas state
        current_elements = st.session_state.canvas.get_elements()
        canvas_state_str = json.dumps(current_elements, ensure_ascii=False)
        dynamic_system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CURRENT CANVAS STATE:\n{canvas_state_str}\n"
            f"IMPORTANT: Preserve existing elements when modifying or extending the diagram unless explicitly asked to remove them."
        )

        if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
            st.session_state.messages[0]["content"] = dynamic_system_prompt
        else:
            st.session_state.messages.insert(0, {"role": "system", "content": dynamic_system_prompt})

        # Run the agent and get its response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                tool_map, tools_schema = make_diagram_tools(st.session_state.canvas, interactive=False)
            except TypeError:
                tool_map, tools_schema = make_diagram_tools(st.session_state.canvas)

            step_count = 0
            while step_count < MAX_STEPS:
                step_count += 1
                compressed_msgs = compress_messages(st.session_state.messages)

                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=compressed_msgs,
                        tools=tools_schema,
                        tool_choice="auto",
                    )

                    response_message = response.choices[0].message

                    if response_message.content:
                        message_placeholder.markdown(response_message.content)
                        st.session_state.messages.append({"role": "assistant", "content": response_message.content})

                    if response_message.tool_calls:
                        st.session_state.messages.append(response_message)
                        for tool_call in response_message.tool_calls:
                            fname = tool_call.function.name
                            fargs = json.loads(tool_call.function.arguments or "{}")

                            st.info(f"⚙️ Running tool: `{fname}`")

                            if fname in tool_map:
                                result = tool_map[fname](**fargs) if callable(tool_map[fname]) else "Tool executed"
                            else:
                                result = f"Error: Tool {fname} not found"

                            st.session_state.messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result)
                            })
                    else:
                        break

                except APIError as e:
                    st.error(f"API Error: {e.message}")
                    break
                except Exception as e:
                    st.error(f"Unexpected Error: {str(e)}")
                    break

        # Rerun the page to update the left column (canvas)
        st.rerun()

# ----------------- Left column: SVG canvas display -----------------
with col_left:
    st.subheader("🎨 Live Diagram Canvas")
    svg_file_path = "output_canvas.svg"

    if os.path.exists(svg_file_path):
        with open(svg_file_path, "r", encoding="utf-8") as f:
            svg_code = f.read()

        # Render the SVG output in the browser
        st.components.v1.html(
            f'<div style="background-color: white; padding: 10px; border-radius: 8px;">{svg_code}</div>',
            height=600,
            scrolling=True
        )
    else:
        st.info("No diagram has been drawn yet. Type your request in the chat box!")
