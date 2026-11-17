# AI Engineering: building LLM agents in Python

A hands-on learning project where I build LLM agents step by step, from a minimal tool-calling agent to agents that generate and evaluate diagrams. Each session adds one engineering concern on top of the previous one: tools, memory, human confirmation, evaluation, context compression, error handling, and streaming.

## What is inside

### Session1: agent fundamentals
- `agent_1.py`: a basic agent loop with tool calling.
- `agent_2_memory.py`: conversation memory.
- `agent_3_tools.py`: multiple tools.
- `agent_4_confirmation.py`: asking the user to confirm before a sensitive action (human in the loop).
- `agent_5_eval.py` and `eval.py`: evaluating agent behaviour.
- `agent_6_compression.py`: compressing long context.
- `agent_7_errorhandling.py`: handling tool and model errors.
- `tools.py` and `test_tools.py`: the tool implementations and tests for them.

### Session2: diagram agents and evaluation
- Agents that draw on a canvas and generate SVG (`agent_1_canvas.py`, `agent_2_svg.py`), a streaming version (`agent_3_streaming.py`), an evaluation agent (`agent_4_eval.py`) and a context-compression agent (`agent_5_compression.py`).
- `golden_evaluation_dataset.json` and `eval.py`: a golden dataset used to score the agents.

### Session3 and Session4: Excalidraw diagram agent
- An agent that creates Excalidraw diagrams through tools (`tools_excalidraw.py`, `tools_excalidraw_v2.py`), with an improvement step (`agent_4_improvement.py`) and an evaluation suite in `evals/`.
- Sample outputs are included as `.excalidraw` files.

## Running the examples

1. Create a Python virtual environment.
2. Install the packages the script you want to run imports (see the imports at the top of each file).
3. Create a `.env` file with your LLM API key and keep it out of version control.
4. Run a script, for example `python Session1/agent_1.py`.

## Notes

This repository is a learning log, so the code favours clarity over production hardening. Next steps I plan: a requirements file, a runnable end-to-end demo, and a short write-up of the evaluation results.
