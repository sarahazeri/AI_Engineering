import streamlit as st
import json
import os
import sys
from openai import OpenAI, APIError
from dotenv import load_dotenv
from tools import Canvas, make_diagram_tools

# 1. initial settings of the Streamlit page
st.set_page_config(
    page_title="Diagram Agent Studio",
    page_icon="🎨",
    layout="wide"
)

# 2. load environment variables and API client
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are an expert Diagram Agent. Your task is to turn user prompts into structured diagram elements.
- Always generate unique element IDs (e.g., 'rect_signup', 'node_validate', 'arrow_1').
- For 'arrow' elements, both 'from' and 'to' fields are STRICTLY REQUIRED and MUST be valid element IDs. Do NOT pass null.
- For nodes (rectangle, diamond, circle), always provide a string 'label'. Do NOT pass null.
- Use `generate_diagram` when creating a brand new diagram from scratch.
- Use `modify_diagram` when updating properties of an existing node or changing its attributes.
- Use `add_elements` when inserting new nodes or decision branches without redrawing existing ones.
"""

MAX_STEPS = 10
MAX_MESSAGES = 12

def compress_messages(messages: list, max_history: int = MAX_MESSAGES) -> list:
    if len(messages) <= max_history:
        return messages
    
    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    recent_messages = messages[-max_history:]
    
    if system_msg and recent_messages[0].get("role") != "system":
        return [system_msg] + recent_messages
    
    return recent_messages

# 3. manage project state (Session State)
if "canvas" not in st.session_state:
    st.session_state.canvas = Canvas("canvas.svg")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. function to run the agent compatible with UI
def run_agent_loop_ui(user_input: str):
    canvas = st.session_state.canvas
    messages = st.session_state.messages
    
    messages.append({"role": "user", "content": user_input})
    
    current_elements = canvas.get_elements()
    canvas_state_str = json.dumps(current_elements, ensure_ascii=False)
    
    dynamic_system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CURRENT CANVAS STATE:\n{canvas_state_str}\n"
        f"IMPORTANT: Preserve existing elements when modifying or extending the diagram unless explicitly asked to remove them."
    )
    
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = dynamic_system_prompt
    else:
        messages.insert(0, {"role": "system", "content": dynamic_system_prompt})

    try:
        tool_map, tools_schema = make_diagram_tools(canvas, interactive=False)
    except TypeError:
        tool_map, tools_schema = make_diagram_tools(canvas)

    step_count = 0

    with st.spinner("agent is analyzing and drawing the diagram..."):
        while step_count < MAX_STEPS:
            step_count += 1
            compressed_msgs = compress_messages(messages)

            try:
                response_stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=compressed_msgs,
                    tools=tools_schema,
                    tool_choice="auto",
                    stream=True
                )

                tool_calls_dict = {}
                assistant_content = ""

                for chunk in response_stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    if delta.content:
                        assistant_content += delta.content

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            index = tc.index
                            if index not in tool_calls_dict:
                                tool_calls_dict[index] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                tool_calls_dict[index]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls_dict[index]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls_dict[index]["arguments"] += tc.function.arguments

                if tool_calls_dict:
                    formatted_tool_calls = []
                    for idx, tc_data in tool_calls_dict.items():
                        formatted_tool_calls.append({
                            "id": tc_data["id"],
                            "type": "function",
                            "function": {
                                "name": tc_data["name"],
                                "arguments": tc_data["arguments"]
                            }
                        })

                    messages.append({
                        "role": "assistant",
                        "content": assistant_content if assistant_content else None,
                        "tool_calls": formatted_tool_calls
                    })

                    for tc_data in tool_calls_dict.values():
                        fname = tc_data["name"]
                        try:
                            fargs = json.loads(tc_data["arguments"])
                        except json.JSONDecodeError:
                            fargs = {}

                        if fname in tool_map:
                            result = tool_map[fname](**fargs)
                        else:
                            result = f"Error: Tool '{fname}' not recognized."

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_data["id"],
                            "content": str(result)
                        })
                else:
                    if assistant_content:
                        messages.append({"role": "assistant", "content": assistant_content})
                    break

            except APIError as e:
                st.error(f"API error: {e.message}")
                break
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                break

# 5. create a two-part user interface (Split Screen)
col_left, col_right = st.columns([1.2, 1])

# --- left side: canvas ---
with col_left:
    st.subheader("🖼️ canvas state (Diagram Canvas)")
    svg_file_path = "canvas.svg"
    
    if os.path.exists(svg_file_path):
        with open(svg_file_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        st.image(svg_file_path, use_container_width=True)
    else:
        st.info("No diagram has been drawn yet. Enter your request in the chat box.")

# --- right side: chat box ---
with col_right:
    st.subheader("💬 chat with the agent")
    
    # display the message history
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            if msg.get("role") == "user":
                st.chat_message("user").write(msg["content"])
            elif msg.get("role") == "assistant" and msg.get("content"):
                st.chat_message("assistant").write(msg["content"])

    # get new input from the user
    if user_prompt := st.chat_input("Request a diagram (e.g., create a signup diagram)..."):
        run_agent_loop_ui(user_prompt)
        st.rerun()