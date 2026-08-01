import os
import sys
import json
from openai import OpenAI, APIError
from dotenv import load_dotenv
from tools_svg import Canvas, make_diagram_tools

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are an expert Diagram Agent. Your task is to turn user prompts into structured diagram elements.
- Always generate unique element IDs (e.g., 'rect_signup', 'node_validate', 'arrow_1').
- Use `generate_diagram` when creating a brand new diagram.
- Use `modify_diagram` when updating properties (e.g. color, label) of an existing node.
- Use `add_elements` when inserting new branches or decision outputs without redrawing the whole canvas.
- When adding decision branches, use labeled arrows (e.g. label='Yes', label='No').
"""

MAX_STEPS = 10

def run_agent_loop(messages: list, canvas: Canvas, interactive: bool = True):
    tool_map, tools_schema = make_diagram_tools(canvas, interactive=interactive)
    step_count = 0

    while step_count < MAX_STEPS:
        step_count += 1
        
        try:
            response_stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                stream=True
            )
        except Exception as e:
            print(f"\n[Agent Error]: {e}")
            break

        tool_calls_dict = {}
        assistant_content = ""

        for chunk in response_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Stream assistant text response
            if delta.content:
                assistant_content += delta.content
                sys.stdout.write(delta.content)
                sys.stdout.flush()

            # Catch tool call announcement
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index
                    if index not in tool_calls_dict:
                        tool_calls_dict[index] = {
                            "id": "",
                            "name": "",
                            "arguments": ""
                        }
                    if tc.id:
                        tool_calls_dict[index]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_dict[index]["name"] = tc.function.name
                        # Announce tool call immediately
                        print(f"\nagent > ⚙️  {tc.function.name}...", end="", flush=True)
                    if tc.function and tc.function.arguments:
                        tool_calls_dict[index]["arguments"] += tc.function.arguments

        # Construct final assistant message
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

            # Execute tool calls
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

                print(f"\nagent > 📋 {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_data["id"],
                    "content": str(result)
                })
        else:
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            break

def start_interactive_cli():
    canvas = Canvas("canvas.svg")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=== Diagram Agent CLI Started ===")
    print("Commands: 'exit' to quit | 'canvas' to view elements\n")

    while True:
        try:
            user_input = input("\nyou > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "canvas":
            print(json.dumps(canvas.get_elements(), indent=2))
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        run_agent_loop(messages, canvas, interactive=True)

if __name__ == "__main__":
    start_interactive_cli()