import os
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Replace with your base_url if needed
    api_key=os.environ.get("GROQ_API_KEY")      # Replace with your API key
)

def start_chat_agent():
    # 1. Initialize message history outside the loop to keep memory
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    
    print("=== AI Agent Chat Started (type 'exit' or 'quit' to stop) ===")
    
    # 2. Infinite loop for continuous conversation
    while True:
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Append user message to history
        messages.append({"role": "user", "content": user_input})
        
        # 3. Call the model
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        
        # Extract assistant response
        assistant_message = response.choices[0].message
        
        # Append model response to history so it remembers next turn
        messages.append(assistant_message)
        
        # Print output
        print(f"\nAgent: {assistant_message.content}")

if __name__ == "__main__":
    start_chat_agent()