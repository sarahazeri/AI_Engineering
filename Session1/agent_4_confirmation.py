import os
import json
from openai import OpenAI
from tools import TOOLS_SCHEMA, TOOL_MAP
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to local tools.
- Answer general knowledge questions directly using your own knowledge without calling tools.
- Only call tools when you explicitly need to read/write/edit files, run shell commands, or search the web.
"""

# Tools that require explicit human approval before execution
DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command"}

def start_agent():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("=== AI Agent Started with Human-in-the-Loop (type 'exit' to quit) ===")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"\n[API Error]: {e}")
            continue
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Check if the model wants to call a tool
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                print(f"\n[Agent Decision]: Wants to call '{function_name}' with args: {arguments}")
                
                # --- Human-In-The-Loop Approval Gate ---
                if function_name in DANGEROUS_TOOLS:
                    approval = input(f"⚠️ [Security Gate] Allow execution of '{function_name}'? (y/n): ").strip().lower()
                    if approval != 'y':
                        result = "User denied this action."
                        print("[Gate Result]: Action DENIED by user.")
                    else:
                        tool_function = TOOL_MAP[function_name]
                        result = tool_function(**arguments)
                        print(f"[Tool Result]: {result}")
                else:
                    # Safe tools execute automatically
                    tool_function = TOOL_MAP[function_name]
                    result = tool_function(**arguments)
                    print(f"[Tool Result]: {result}")
                
                # Append result to memory
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
                
                # Return tool result back to model
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages
                )
                final_message = final_response.choices[0].message
                messages.append(final_message)
                print(f"\nAgent: {final_message.content}")
        else:
            print(f"\nAgent: {response_message.content}")

if __name__ == "__main__":
    start_agent()