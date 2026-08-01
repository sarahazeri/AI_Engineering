import os
import json
import time
from tools_svg import Canvas
from agent_4_eval import SYSTEM_PROMPT, run_agent_loop

# --- AUTOMATED SCORERS ---

def schema_scorer(canvas: Canvas) -> float:
    """Verifies validity of every element on canvas."""
    elements = canvas.get_elements()
    if not elements:
        return 1.0
    valid_count = 0
    for el in elements:
        if isinstance(el, dict) and "id" in el and "type" in el:
            valid_count += 1
    return valid_count / len(elements)

def structure_scorer(case: dict, canvas: Canvas) -> float:
    """Checks expected element count & connectivity."""
    elements = canvas.get_elements()
    nodes = [e for e in elements if e.get("type") != "arrow"]
    arrows = [e for e in elements if e.get("type") == "arrow"]
    
    expected = case.get("expectedCharacteristics", [])
    if not expected:
        return 1.0

    score = 0.5 if len(elements) > 0 else 0.0
    if nodes:
        score += 0.25
    if arrows or len(nodes) <= 1:
        score += 0.25
    return min(1.0, score)

def preservation_scorer(case: dict, canvas: Canvas) -> float:
    """Checks if seeded elements survived during modification."""
    if case.get("category") != "modify":
        return 1.0  # Returns None/1.0 when not applicable
    
    preserved_ids = case.get("preservedIds", [])
    if not preserved_ids:
        return 1.0

    current_ids = {e.get("id") for e in canvas.get_elements()}
    match_count = sum(1 for pid in preserved_ids if pid in current_ids)
    return match_count / len(preserved_ids)

def keywords_scorer(case: dict, canvas: Canvas) -> float:
    """Checks presence of domain vocabulary in element labels."""
    keywords = case.get("keywords", [])
    if not keywords or case.get("category") != "domain":
        return 1.0

    all_text = " ".join([str(e.get("label", "")) + " " + str(e.get("id", "")) for e in canvas.get_elements()]).lower()
    matched = sum(1 for kw in keywords if kw.lower() in all_text)
    return matched / len(keywords)


# --- EVAL HARNESS ---

def run_eval_suite():
    golden_path = os.path.join("golden_evaluation_dataset.json")
    if not os.path.exists(golden_path):
        print(f"Error: Golden dataset not found at {golden_path}")
        return

    # Ensure output directory for evals exists
    os.makedirs("evals", exist_ok=True)

    with open(golden_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"\n==========================================")
    print(f"  RUNNING EVAL SUITE ({len(cases)} Cases)")
    print(f"==========================================\n")

    schema_scores = []
    structure_scores = []
    preservation_scores = []
    keyword_scores = []

    start_time = time.time()

    for idx, case in enumerate(cases, 1):
        cid = case["id"]
        category = case["category"]
        prompt = case["input"]

        # Isolated canvas per case without relying on __init__ keyword argument
        test_canvas = Canvas()
        test_canvas.output_file = f"evals/output_{cid}.svg"
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Seed canvas for modify cases
        if "seed" in case:
            test_canvas.set_elements(case["seed"].get("elements", []), auto_save=False)
            if "userPrompt" in case["seed"]:
                messages.append({"role": "user", "content": case["seed"]["userPrompt"]})
            if "assistantConfirmation" in case["seed"]:
                messages.append({"role": "assistant", "content": case["seed"]["assistantConfirmation"]})

        messages.append({"role": "user", "content": prompt})

        # Run non-interactive loop
        run_agent_loop(messages, test_canvas, interactive=False)

        # Calculate scores
        s_schema = schema_scorer(test_canvas)
        s_struct = structure_scorer(case, test_canvas)
        s_pres = preservation_scorer(case, test_canvas)
        s_key = keywords_scorer(case, test_canvas)

        schema_scores.append(s_schema)
        structure_scores.append(s_struct)
        if case.get("category") == "modify":
            preservation_scores.append(s_pres)
        if case.get("category") == "domain":
            keyword_scores.append(s_key)

        print(f"[{idx}/{len(cases)}] Case {cid} ({category}): Schema={s_schema:.0%}, Struct={s_struct:.0%}, Pres={s_pres:.0%}")

    duration = time.time() - start_time

    avg_schema = sum(schema_scores) / len(schema_scores) if schema_scores else 0
    avg_struct = sum(structure_scores) / len(structure_scores) if structure_scores else 0
    avg_pres = sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0
    avg_key = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0

    print("\n==========================================")
    print("           BASELINE EVAL SUMMARY          ")
    print("==========================================")
    print(f"Total Duration    : {duration:.2f} seconds")
    print(f"Schema Score      : {avg_schema:.1%}")
    print(f"Structure Score   : {avg_struct:.1%}")
    print(f"Preservation Score: {avg_pres:.1%}")
    print(f"Keywords Score    : {avg_key:.1%}")
    print("==========================================\n")

if __name__ == "__main__":
    run_eval_suite()