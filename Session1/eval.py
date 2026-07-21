import sys
import os

from agent_5_eval import run_agent_turn

EVAL_CASES = [
    {
        "name": "General Knowledge (No tool expected)",
        "prompt": "What is the capital of France?",
        "expected_tool": None
    },
    {
        "name": "List Files Execution",
        "prompt": "List all files in the current folder.",
        "expected_tool": "list_files"
    },
    {
        "name": "File Writing Request",
        "prompt": "Create a file named test_eval.txt with content 'test'",
        "expected_tool": "write_file"
    },
    {
        "name": "Command Execution Request",
        "prompt": "Run a shell command to show python version",
        "expected_tool": "run_command"
    }
]

def run_evals():
    print("=== Running Evaluation Suite on Your Agent ===\n")
    passed = 0
    total = len(EVAL_CASES)
    
    for test in EVAL_CASES:
        print(f"Testing Prompt: '{test['prompt']}'")
        called_tool, response_text = run_agent_turn(test["prompt"])
        
        if called_tool == test["expected_tool"]:
            print(f"✅ PASS | Expected Tool: {test['expected_tool']} | Agent Called: {called_tool}\n")
            passed += 1
        else:
            print(f"❌ FAIL | Expected Tool: {test['expected_tool']} | Agent Called: {called_tool}\n")
            
    pass_rate = (passed / total) * 100
    print("=" * 50)
    print(f"EVAL SUMMARY: {passed}/{total} passed ({pass_rate:.1f}% Pass Rate)")
    print("=" * 50)

if __name__ == "__main__":
    run_evals()