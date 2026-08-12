import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, APIError
from tools_excalidraw import ExcalidrawCanvas
from tools_excalidraw_v2 import make_diagram_tools

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Diagram Agent - Excalidraw v2", layout="wide")
st.title("📊 Diagram Agent - Chatbot & Excalidraw Canvas (v2)")

# 1. Initialize Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are an expert Diagram Agent that produces Excalidraw system-design diagrams.
- Always generate unique node IDs (e.g., 'node_signup', 'node_validate').
- For nodes, always provide 'id', 'label' and 'shape' ('rectangle', 'diamond', or 'ellipse').
- Give a node an 'icon' (a single emoji, e.g. a server/gear/mail/database glyph) whenever it helps
  communicate what the node represents at a glance - this is expected for system-design/architecture
  diagrams, not just plain flowcharts.
- Use 'groups' to draw a titled container box around a set of related nodes, e.g. a subsystem made of
  several internal components (like a "Resource Manager" holding several queues + a scheduler), or a
  pool of interchangeable workers (like a "Task Workers" box holding several worker icons with no
  relationships between them - those will automatically be arranged in a grid). Reference a group's id
  from a node's 'group' field to place that node inside it. Use 'style': 'dashed' for a looser/optional
  pool and 'solid' for a concrete subsystem boundary.
- An arrow's 'from'/'to' can reference either a specific node ID, or a group ID directly if the arrow
  should point at the whole container (e.g. a "run task" arrow from a scheduler node to a workers GROUP,
  not to one specific worker).
- For arrows, both 'from' and 'to' fields are STRICTLY REQUIRED and MUST reference a valid node ID or
  group ID.
- Layout is fully automatic: columns are derived from the arrow graph, branches/queues/groups fan out
  into rows or a grid as needed, and long or skip-connection arrows are routed around other nodes
  instead of through them. You don't need to specify positions or worry about crossings - just describe
  the nodes, their groups, and how they connect.
- Use generate_diagram to (re)create the full diagram from scratch with the complete set of
  nodes/arrows/groups; it always replaces the previous diagram, so when the user asks to extend or
  modify the current diagram, include ALL previous nodes/arrows/groups (from CURRENT CANVAS STATE)
  plus the new/changed ones in the same call.
"""

MAX_STEPS = 10
MAX_MESSAGES = 12

# 2. Manage Session State to persist chat history and canvas
if "canvas" not in st.session_state:
    st.session_state.canvas = ExcalidrawCanvas("diagram_v2.excalidraw")

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

# 4. Render the real Excalidraw canvas (embedded from CDN) as standalone HTML
def _diagram_bounds(elements: list) -> dict:
    """Bounding box of all elements, in canvas (world) coordinates."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for el in elements:
        if "x" not in el or "y" not in el:
            continue
        w, h = el.get("width", 0) or 0, el.get("height", 0) or 0
        x0, x1 = el["x"], el["x"] + w
        y0, y1 = el["y"], el["y"] + h
        min_x, max_x = min(min_x, x0, x1), max(max_x, x0, x1)
        min_y, max_y = min(min_y, y0, y1), max(max_y, y0, y1)
    if min_x == float("inf"):
        return {"minX": 0, "maxX": 0, "minY": 0, "maxY": 0}
    return {"minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y}


def render_excalidraw(elements: list, height: int = 650) -> str:
    elements_json = json.dumps(elements, ensure_ascii=False)
    bounds_json = json.dumps(_diagram_bounds(elements))
    return f"""
    <div id="app-root" style="width:100%; height:{height}px; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden;"></div>
    <script src="https://unpkg.com/react@18.3.1/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js"></script>
    <script>
      window.EXCALIDRAW_ASSET_PATH = "https://unpkg.com/@excalidraw/excalidraw@0.17.6/dist/";
    </script>
    <script src="https://unpkg.com/@excalidraw/excalidraw@0.17.6/dist/excalidraw.production.min.js"></script>
    <script>
      const container = document.getElementById("app-root");
      try {{
        // Anchor the diagram's right edge to the right side of the pane at
        // full zoom (1:1) instead of letting it default to a centered,
        // zoomed-to-fit view - keeps text readable and matches RTL reading.
        const bounds = {bounds_json};
        const PADDING = 60;
        const vw = container.clientWidth || {height};
        const vh = container.clientHeight || {height};
        const scrollX = (vw - PADDING) - bounds.maxX;
        const scrollY = (vh / 2) - ((bounds.minY + bounds.maxY) / 2);
        const root = ReactDOM.createRoot(container);
        root.render(
          React.createElement(ExcalidrawLib.Excalidraw, {{
            initialData: {{
              elements: {elements_json},
              appState: {{ viewBackgroundColor: "#ffffff", scrollX: scrollX, scrollY: scrollY, zoom: {{ value: 1 }} }}
            }}
          }})
        );
      }} catch (e) {{
        container.innerHTML = "<p style='padding:16px;font-family:sans-serif;color:#b91c1c;'>Failed to load the Excalidraw canvas. Check your internet connection.</p>";
      }}
    </script>
    """

# 5. Build the two-column layout (Split Screen)
col_canvas, col_chat = st.columns([1.2, 1])

# ----------------- Right column: Chatbot -----------------
with col_chat:
    st.subheader("💬 Chat with the Agent")

    for msg in st.session_state.messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

        if role in ["user", "assistant"] and content:
            with st.chat_message(role):
                st.write(content)

    user_input = st.chat_input("Describe the diagram you want...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        current_elements = st.session_state.canvas.get_elements()
        canvas_state_str = json.dumps(current_elements, ensure_ascii=False)
        dynamic_system_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CURRENT CANVAS STATE (raw Excalidraw elements):\n{canvas_state_str}\n"
        )

        if st.session_state.messages and st.session_state.messages[0].get("role") == "system":
            st.session_state.messages[0]["content"] = dynamic_system_prompt
        else:
            st.session_state.messages.insert(0, {"role": "system", "content": dynamic_system_prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            tool_map, tools_schema = make_diagram_tools(st.session_state.canvas, interactive=False)

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

                    if response_message.tool_calls:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                                }
                                for tc in response_message.tool_calls
                            ]
                        })
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
                        if response_message.content:
                            st.session_state.messages.append({"role": "assistant", "content": response_message.content})
                        break

                except APIError as e:
                    st.error(f"API Error: {e.message}")
                    break
                except Exception as e:
                    st.error(f"Unexpected Error: {str(e)}")
                    break

        st.rerun()

# ----------------- Left column: Excalidraw canvas display -----------------
with col_canvas:
    st.subheader("🎨 Live Diagram Canvas (Excalidraw)")
    elements = st.session_state.canvas.get_elements()

    if elements:
        st.components.v1.html(render_excalidraw(elements), height=670, scrolling=False)
    else:
        st.info("No diagram has been drawn yet. Type your request in the chat box!")
