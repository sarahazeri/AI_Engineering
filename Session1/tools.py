import os
import subprocess

# --- TOOL IMPLEMENTATIONS ---

def read_file(path: str) -> str:
    """Reads and returns the full content of a file at the specified path."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file '{path}': {e}"

def write_file(path: str, content: str) -> str:
    """Writes content to a file. Overwrites the file if it already exists."""
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing to file '{path}': {e}"

def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replaces occurrences of old_str with new_str in the specified file."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if old_str not in content:
            return f"Error: target string '{old_str}' not found in {path}"
            
        updated_content = content.replace(old_str, new_str)
        
        with open(path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
            
        return f"Successfully updated {path}"
    except Exception as e:
        return f"Error editing file '{path}': {e}"

def list_files(directory: str = ".") -> str:
    """Lists all files and directories in the given directory path."""
    try:
        files = os.listdir(directory)
        return "\n".join(files) if files else "Directory is empty."
    except Exception as e:
        return f"Error listing directory '{directory}': {e}"

def run_command(command: str) -> str:
    """Executes a shell command and returns combined stdout and stderr."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        return output.strip() if output.strip() else "Command executed with no output."
    except Exception as e:
        return f"Error executing command '{command}': {e}"

def web_search(query: str) -> str:
    """Simulates a web search query to retrieve recent information."""
    return f"Search result for '{query}': No live search API configured."

# --- TOOL SCHEMAS FOR OPENAI API ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The target file path."},
                    "content": {"type": "string", "description": "The content to write."}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific substring in a file with a new string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The target file path."},
                    "old_str": {"type": "string", "description": "Text to be replaced."},
                    "new_str": {"type": "string", "description": "Replacement text."}
                },
                "required": ["path", "old_str", "new_str"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path (defaults to current directory '.')."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a bash shell command on the host system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"]
            }
        }
    }
]

# Mapping tool names to executable python functions
TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
    "run_command": run_command,
    "web_search": web_search
}