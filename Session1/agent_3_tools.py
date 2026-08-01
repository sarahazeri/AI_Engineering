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

# Added a clear system instruction to prevent tool formatting errors
SYSTEM_PROMPT = """You are a helpful AI assistant with access to local tools.
- Answer general knowledge questions directly using your own knowledge without calling tools.
- Only call tools when you explicitly need to read/write/edit files, run shell commands, or search the web.
"""

def start_agent():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("=== AI Agent Started (type 'exit' to quit) ===")
    
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
                
                # Safely parse arguments
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                print(f"\n[Agent Decision]: Calling tool '{function_name}' with args: {arguments}")
                
                if function_name in TOOL_MAP:
                    tool_function = TOOL_MAP[function_name]
                    result = tool_function(**arguments)
                    
                    print(f"[Tool Result]: {result}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                    
                    # Send tool result back to model
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