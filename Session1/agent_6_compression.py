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

SYSTEM_PROMPT = """You are a helpful AI assistant.
1. Answer general knowledge questions directly using your own knowledge without calling any tool.
2. Only call tools when strictly necessary.
3. When calling a function, output ONLY valid JSON arguments matching the schema."""

DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command"}

def compress_messages(messages: list, max_messages: int = 8) -> list:
    """
    Function to compress messages to keep the context window within the model's limit.
    """
    if len(messages) <= max_messages:
        return messages
    
    print("\n⚡ [Compression]: Context window is getting large. Compressing old messages...")
    
    # Separate system message (index 0)
    system_msg = messages[0]
    # Old messages that need to be compressed
    old_messages = messages[1:-4]
    # Recent messages that should remain unchanged
    recent_messages = messages[-4:]
    
    # Request from model to summarize old messages
    summary_prompt = [
        {"role": "system", "content": "Summarize the key information and facts from this conversation history into a concise summary."},
        {"role": "user", "content": json.dumps([m if isinstance(m, dict) else m.model_dump() for m in old_messages])}
    ]
    
    try:
        summary_response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=summary_prompt
        )
        summary_text = summary_response.choices[0].message.content
        
        # Create new compressed history
        compressed_history = [
            system_msg,
            {"role": "system", "content": f"Previous conversation summary: {summary_text}"}
        ] + recent_messages
        
        print("✅ [Compression]: Messages successfully compressed!")
        return compressed_history
    except Exception as e:
        print(f"⚠️ [Compression Failed]: {e}. Fallback to truncation.")
        # In case of error, Truncation (simple deletion) is applied
        return [system_msg] + recent_messages

def run_agent_turn(user_prompt: str):
    """Single turn execution for Evals or API calls"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )
        message = response.choices[0].message
        called_tool = None
        if message.tool_calls:
            called_tool = message.tool_calls[0].function.name
        return called_tool, message.content
    except Exception as e:
        return None, f"Error: {e}"

def start_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=== AI Agent Started with Human-in-the-Loop & Compression (type 'exit' to quit) ===")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        # Apply compression function before sending request to API
        messages = compress_messages(messages)
        
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
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                # 📌 Section to display tool details in console
                print(f"\n🛠️  [TOOL EXECUTING]: Agent selected tool -> '{function_name}'")
                print(f"📋 [Arguments]: {json.dumps(arguments, ensure_ascii=False)}")
                
                if function_name in DANGEROUS_TOOLS:
                    approval = input(f"⚠️ [Security Gate] Allow execution of '{function_name}'? (y/n): ").strip().lower()
                    if approval != 'y':
                        result = "User denied this action."
                        print("[Gate Result]: Action DENIED by user.")
                    else:
                        tool_function = TOOL_MAP[function_name]
                        result = tool_function(**arguments)
                        print(f"[Tool Output]: {result}")
                else:
                    tool_function = TOOL_MAP[function_name]
                    result = tool_function(**arguments)
                    print(f"[Tool Output]: {result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
                
                # Recompression before final response
                messages = compress_messages(messages)
                
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