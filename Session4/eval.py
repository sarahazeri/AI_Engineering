"""
Eval suite for the Excalidraw v2 diagram agent (agent_4_improvement.py).

Unlike a smoke test that just checks "did the canvas get some elements",
every case here targets a specific behavior the agent's system prompt
promises and a weak/small model can plausibly get wrong:

  - extend-fidelity : does a follow-up turn keep ALL prior nodes/arrows
                       (the system prompt explicitly demands this), and
                       does it actually rewire things rather than just
                       bolting a new node on?
  - group-semantics  : dashed/unwired pool vs. solid/wired subsystem,
                       arrows targeting a group id instead of a member.
  - adversarial      : contradictory or malformed requests (dual group
                       membership, invalid shape, dangling reference)
                       that the schema/tool does NOT protect against by
                       itself - only sane model behavior does.
  - icon-judgment    : icons expected for architecture diagrams, but the
                       model shouldn't over-engineer a trivial flowchart
                       (extra groups, wrong step count).
  - scale/topology   : exact node/arrow counts on a longer spec, and a
                       genuine cycle (retry loop) that also exercises the
                       layering code's cycle handling.

Each case captures the RAW arguments the model passed to generate_diagram
(before layout) for semantic checks, plus the final rendered canvas (after
layout) for two structural integrity checks that apply to every case:
dangling arrow bindings and overlapping sibling shapes.

Scores are intentionally strict (exact counts, exact wiring, not "contains
plausible-looking text") - a case that "sort of" satisfies the prompt
should not read as a pass.
"""
import os
import json
import time
import braintrust
from dotenv import load_dotenv
from openai import OpenAI, APIError

from tools_excalidraw import ExcalidrawCanvas
from tools_excalidraw_v2 import make_diagram_tools

load_dotenv()

# Braintrust: logs every model call made by the eval suite as a trace under this project.
braintrust.init_logger(project="My Project")

client = braintrust.wrap_openai(OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
))

MODEL = os.environ.get("GROQ_EVAL_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = 10

# Kept in sync with Session3/agent_4_improvement.py SYSTEM_PROMPT.
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


# --- Harness -----------------------------------------------------------

def run_turn(messages: list, canvas: ExcalidrawCanvas) -> dict | None:
    """Runs one tool-calling loop for a turn whose user message has
    already been appended to `messages`. Returns the raw kwargs
    (nodes/arrows/groups) of the LAST generate_diagram call the model
    made this turn (pre-layout), or None if it never called the tool."""

    dynamic_system_prompt = (
        f"{SYSTEM_PROMPT}\n\nCURRENT CANVAS STATE (raw Excalidraw elements):\n"
        f"{json.dumps(canvas.get_elements(), ensure_ascii=False)}\n"
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = dynamic_system_prompt
    else:
        messages.insert(0, {"role": "system", "content": dynamic_system_prompt})

    tool_map, tools_schema = make_diagram_tools(canvas, interactive=False)
    real_generate = tool_map["generate_diagram"]
    captured = {}

    def spy(**kwargs):
        captured["nodes"] = kwargs.get("nodes") or []
        captured["arrows"] = kwargs.get("arrows") or []
        captured["groups"] = kwargs.get("groups") or []
        return real_generate(**kwargs)

    tool_map["generate_diagram"] = spy

    for _ in range(MAX_STEPS):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools_schema, tool_choice="auto",
            )
        except APIError as e:
            print(f"    [API error mid-turn: {e}]")
            break

        rm = response.choices[0].message
        if rm.tool_calls:
            messages.append({
                "role": "assistant",
                "content": rm.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in rm.tool_calls
                ],
            })
            for tc in rm.tool_calls:
                fname = tc.function.name
                try:
                    fargs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fargs = {}
                result = tool_map[fname](**fargs) if fname in tool_map else f"Error: tool {fname} not found"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
        else:
            if rm.content:
                messages.append({"role": "assistant", "content": rm.content})
            break

    return captured if "nodes" in captured else None


# --- Shared helpers for check functions --------------------------------

def node_by_kw(nodes, *kws):
    for n in nodes:
        hay = f"{n.get('label', '')} {n.get('id', '')}".lower()
        if all(kw.lower() in hay for kw in kws):
            return n
    return None


def arrow_set(arrows):
    return {(a.get("from"), a.get("to")) for a in arrows}


def dangling_refs(canvas) -> list:
    """Every arrow's start/end binding must resolve to a real, non-deleted
    element in the FINAL rendered canvas - a real end-to-end integrity
    check, not just a check on what the model claimed."""
    ids = {e["id"] for e in canvas.get_elements() if not e.get("isDeleted")}
    bad = []
    for e in canvas.get_elements():
        if e.get("type") == "arrow":
            sb = (e.get("startBinding") or {}).get("elementId")
            eb = (e.get("endBinding") or {}).get("elementId")
            if sb not in ids or eb not in ids:
                bad.append(e["id"])
    return bad


def overlapping_pairs(canvas, group_ids: set) -> list:
    """Sibling leaf shapes (not group container boxes) whose bounding
    boxes overlap by more than a hairline - a real layout regression."""
    leaves = [e for e in canvas.get_elements()
              if e.get("type") in ("rectangle", "diamond", "ellipse") and e["id"] not in group_ids]
    bad = []
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            a, b = leaves[i], leaves[j]
            ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
            bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
            ox = max(0, min(ax1, bx1) - max(ax0, bx0))
            oy = max(0, min(ay1, by1) - max(ay0, by0))
            if ox > 2 and oy > 2:
                bad.append((a["id"], b["id"]))
    return bad


# --- Per-case checkers ---------------------------------------------------
# Each returns (score in [0,1], list[str] of notes explaining the score).

def check_ext_01(turns, canvas):
    t1, t2 = turns[0], turns[1]
    if not t1 or not t2:
        return 0.0, ["model never called generate_diagram in one of the two turns"]
    notes = []
    t1_ids = {n["id"] for n in t1["nodes"]}
    t2_ids = {n["id"] for n in t2["nodes"]}
    preserved = t1_ids & t2_ids
    ratio = len(preserved) / len(t1_ids) if t1_ids else 0
    score = 0.5 * ratio
    notes.append(f"preserved {len(preserved)}/{len(t1_ids)} turn-1 node ids in turn 2")

    new_ids = t2_ids - t1_ids
    validate_new = node_by_kw([n for n in t2["nodes"] if n["id"] in new_ids], "valid")
    if validate_new:
        score += 0.3
        notes.append("a new 'Validate' node was added")
    else:
        notes.append(f"no new node resembling 'Validate' found (new ids: {sorted(new_ids)})")

    clean = node_by_kw(t2["nodes"], "clean")
    store = node_by_kw(t2["nodes"], "store")
    aset = arrow_set(t2["arrows"])
    if validate_new and clean and store and (clean["id"], validate_new["id"]) in aset and (validate_new["id"], store["id"]) in aset:
        score += 0.2
        notes.append("Validate correctly rewired between Clean and Store")
    else:
        notes.append("Validate not correctly spliced into the Clean -> Store edge")
    return min(score, 1.0), notes


def check_ext_02(turns, canvas):
    t1, t2 = turns[0], turns[1]
    if not t1 or not t2:
        return 0.0, ["model never called generate_diagram in one of the two turns"]
    notes = []
    t1_ids = {n["id"] for n in t1["nodes"]}
    t2_ids = {n["id"] for n in t2["nodes"]}
    ratio = len(t1_ids & t2_ids) / len(t1_ids) if t1_ids else 0
    score = 0.2 * ratio
    notes.append(f"preserved {len(t1_ids & t2_ids)}/{len(t1_ids)} turn-1 node ids")

    cache = node_by_kw(t2["nodes"], "cache")
    if cache:
        score += 0.2
        notes.append("Redis Cache node created")
    else:
        notes.append("no Cache node found")
        return min(score, 1.0), notes

    app1 = node_by_kw(t2["nodes"], "app1") or node_by_kw(t2["nodes"], "app", "1")
    app2 = node_by_kw(t2["nodes"], "app2") or node_by_kw(t2["nodes"], "app", "2")
    db = node_by_kw(t2["nodes"], "database") or node_by_kw(t2["nodes"], "db")
    aset = arrow_set(t2["arrows"])
    wired = bool(app1 and app2 and db and (app1["id"], cache["id"]) in aset
                 and (app2["id"], cache["id"]) in aset and (cache["id"], db["id"]) in aset)
    if wired:
        score += 0.4
        notes.append("both App Servers route through Cache into Database")
    else:
        notes.append("cache not correctly wired between both App Servers and the Database")

    old_direct_gone = not (app1 and db and (app1["id"], db["id"]) in aset) and \
                       not (app2 and db and (app2["id"], db["id"]) in aset)
    if old_direct_gone:
        score += 0.2
        notes.append("old direct App -> Database edges were removed (real rewire, not just an addition)")
    else:
        notes.append("stale direct App -> Database edge(s) still present alongside the cache")
    return min(score, 1.0), notes


def check_grp_01(turns, canvas):
    t = turns[0]
    if not t:
        return 0.0, ["model never called generate_diagram"]
    notes = []
    score = 0.0
    group = next((g for g in t["groups"] if "worker" in f"{g.get('id','')} {g.get('title','')}".lower()), None)
    if not group:
        return 0.0, ["no 'Task Workers' group produced"]

    if group.get("style") == "dashed":
        score += 0.25
        notes.append("group correctly styled 'dashed' for an interchangeable pool")
    else:
        notes.append(f"expected style 'dashed' for a worker pool, got {group.get('style')!r}")

    members = [n for n in t["nodes"] if n.get("group") == group["id"]]
    if len(members) == 4:
        score += 0.25
        notes.append("exactly 4 worker nodes placed in the group")
    else:
        notes.append(f"expected 4 nodes in the group, got {len(members)}")

    member_ids = {m["id"] for m in members}
    internal = [a for a in t["arrows"] if a.get("from") in member_ids and a.get("to") in member_ids]
    if not internal:
        score += 0.25
        notes.append("no spurious wiring between interchangeable workers")
    else:
        notes.append(f"unexpected internal arrows between workers: {internal}")

    scheduler = node_by_kw(t["nodes"], "schedul")
    targets_group = bool(scheduler and any(a.get("from") == scheduler["id"] and a.get("to") == group["id"] for a in t["arrows"]))
    if targets_group:
        score += 0.25
        notes.append("Scheduler arrow correctly targets the group id, not one specific worker")
    else:
        notes.append("Scheduler does not have an arrow pointing at the group id itself")
    return score, notes


def check_grp_02(turns, canvas):
    t = turns[0]
    if not t:
        return 0.0, ["model never called generate_diagram"]
    notes = []
    score = 0.0
    group = next((g for g in t["groups"] if "resource manager" in f"{g.get('id','')} {g.get('title','')}".lower()), None)
    if not group:
        return 0.0, ["no 'Resource Manager' group produced"]

    if group.get("style") == "solid":
        score += 0.25
        notes.append("group correctly styled 'solid' for a concrete subsystem")
    else:
        notes.append(f"expected style 'solid', got {group.get('style')!r}")

    members = [n for n in t["nodes"] if n.get("group") == group["id"]]
    if len(members) >= 3:
        score += 0.25
        notes.append(f"{len(members)} members placed in the group (scheduler + 2 queues)")
    else:
        notes.append(f"expected >=3 members (scheduler + 2 queues), got {len(members)}")

    scheduler = node_by_kw(members, "schedul")
    member_ids = {m["id"] for m in members}
    internal = [a for a in t["arrows"] if a.get("from") == (scheduler or {}).get("id") and a.get("to") in member_ids]
    if scheduler and len(internal) >= 2:
        score += 0.25
        notes.append("scheduler wired to both queues inside the group")
    else:
        notes.append("scheduler not wired to (both) queues inside the group")

    api = node_by_kw(t["nodes"], "api")
    into_group = bool(api and any(a.get("from") == api["id"] and a.get("to") == group["id"] for a in t["arrows"]))
    if into_group:
        score += 0.25
        notes.append("external API arrow correctly targets the group id")
    else:
        notes.append("API node does not have an arrow pointing at the group id")
    return score, notes


def check_grp_03(turns, canvas):
    t = turns[0]
    if not t:
        return 0.0, ["model never called generate_diagram"]
    notes = []
    auth = node_by_kw(t["nodes"], "auth")
    if not auth:
        return 0.0, ["no Auth node produced"]

    score = 0.0
    group_field = auth.get("group")
    if group_field is None or isinstance(group_field, str):
        score += 0.5
        notes.append(f"'group' field is schema-safe ({group_field!r}) despite the contradictory request")
    else:
        notes.append(f"'group' field is not a plain string/None: {group_field!r} - would break the tool schema")

    final_ids = [e["id"] for e in canvas.get_elements() if e.get("type") in ("rectangle", "diamond", "ellipse")]
    count = final_ids.count(auth["id"])
    if count == 1:
        score += 0.5
        notes.append("Auth node rendered exactly once (not duplicated to satisfy both groups)")
    else:
        notes.append(f"Auth node id appears {count} times in the final canvas")
    return score, notes


def check_adv_01(turns, canvas):
    t1, t2 = turns[0], turns[1]
    if not t1 or not t2:
        return 0.0, ["model never called generate_diagram in one of the two turns"]
    notes = []
    score = 0.0
    legacy = node_by_kw(t2["nodes"], "legacy") or node_by_kw(t2["nodes"], "mainframe")
    if legacy:
        score += 0.5
        notes.append("model created the implied LegacyMainframe node instead of leaving a dangling reference")
    else:
        notes.append("model referenced LegacyMainframe without ever creating the node")

    cache = node_by_kw(t2["nodes"], "cache")
    aset = arrow_set(t2["arrows"])
    if legacy and cache and (cache["id"], legacy["id"]) in aset:
        score += 0.3
        notes.append("arrow correctly wired Cache -> LegacyMainframe")
    else:
        notes.append("Cache -> LegacyMainframe arrow missing or misdirected")

    bad = dangling_refs(canvas)
    if not bad:
        score += 0.2
        notes.append("no dangling arrow bindings in the final rendered canvas")
    else:
        notes.append(f"dangling arrows in final canvas: {bad}")
    return score, notes


def check_adv_02(turns, canvas):
    t = turns[0]
    if not t or not t["nodes"]:
        return 0.0, ["model produced no nodes at all - likely failed on the invalid 'hexagon' shape request"]
    notes = ["model still produced a diagram despite the invalid shape request"]
    score = 0.3
    special = node_by_kw(t["nodes"], "special")
    if special:
        shape = special.get("shape", "rectangle")
        if shape in ("rectangle", "diamond", "ellipse"):
            score += 0.4
            notes.append(f"'hexagon' request degraded gracefully to allowed shape '{shape}'")
        else:
            notes.append(f"shape '{shape}' is outside the allowed enum")
    else:
        notes.append("no 'Special Step' node found")
    normal = node_by_kw(t["nodes"], "normal")
    if normal:
        score += 0.3
        notes.append("'Normal Step' node also present and unaffected")
    return score, notes


def check_icon_01(turns, canvas):
    t = turns[0]
    if not t:
        return 0.0, ["model never called generate_diagram"]
    nodes = t["nodes"]
    if len(nodes) < 4:
        return 0.2, [f"only {len(nodes)} nodes produced for a 6-entity architecture prompt"]
    notes = []
    with_icon = [n for n in nodes if (n.get("icon") or "").strip()]
    ratio = len(with_icon) / len(nodes)
    notes.append(f"{len(with_icon)}/{len(nodes)} nodes carry an icon ({ratio:.0%})")
    icon_score = min(1.0, ratio / 0.7)

    expected_kw = ["client", "gateway", "auth", "payment", "database", "queue"]
    found = sum(1 for kw in expected_kw if node_by_kw(nodes, kw))
    notes.append(f"found {found}/{len(expected_kw)} expected entities from the prompt")
    return 0.6 * icon_score + 0.4 * (found / len(expected_kw)), notes


def check_icon_02(turns, canvas):
    t = turns[0]
    if not t:
        return 0.0, ["model never called generate_diagram"]
    nodes, arrows, groups = t["nodes"], t["arrows"], t.get("groups") or []
    notes = []
    score = 0.0
    if len(nodes) == 4:
        score += 0.4
        notes.append("exactly 4 steps produced")
    else:
        notes.append(f"expected 4 nodes, got {len(nodes)}")

    if len(arrows) == 3:
        score += 0.3
    else:
        notes.append(f"expected 3 arrows, got {len(arrows)}")

    order_kws = ["boil", "steep", "milk", "serve"]
    ordered = [node_by_kw(nodes, kw) for kw in order_kws]
    if all(ordered):
        aset = arrow_set(arrows)
        chain_ok = all((ordered[i]["id"], ordered[i + 1]["id"]) in aset for i in range(3))
        if chain_ok:
            score += 0.2
            notes.append("sequential Boil -> Steep -> Add Milk -> Serve order preserved")
        else:
            notes.append("all 4 steps present but not chained in the requested order")
    else:
        notes.append("not all 4 expected steps found by keyword")

    if not groups:
        score += 0.1
        notes.append("no unnecessary group container introduced for a trivial linear flow")
    else:
        notes.append("model over-engineered a trivial flow with group container(s)")
    return min(score, 1.0), notes


def check_scale_01(turns, canvas):
    t = turns[0]
    if not t:
        return 0.0, ["model never called generate_diagram"]
    nodes, arrows = t["nodes"], t["arrows"]
    notes = []
    score = 0.0
    if len(nodes) == 8:
        score += 0.3
        notes.append("exactly 8 nodes (7 stages + Rollback)")
    else:
        notes.append(f"expected 8 nodes, got {len(nodes)}")

    stage_kws = ["lint", "unit", "build", "integration", "security", "staging", "production", "rollback"]
    found = [node_by_kw(nodes, kw) for kw in stage_kws]
    if all(found):
        score += 0.2
        notes.append("all 8 expected stages present by keyword")
    else:
        notes.append(f"missing expected stages: {[kw for kw, n in zip(stage_kws, found) if not n]}")

    if len(arrows) == 8:
        score += 0.2
    else:
        notes.append(f"expected 8 arrows (6 sequential + 2 into Rollback), got {len(arrows)}")

    if all(found):
        aset = arrow_set(arrows)
        seq_ok = all((found[i]["id"], found[i + 1]["id"]) in aset for i in range(6))
        rollback_ok = (found[5]["id"], found[7]["id"]) in aset and (found[6]["id"], found[7]["id"]) in aset
        if seq_ok:
            score += 0.15
            notes.append("main 7-stage chain correctly sequential")
        else:
            notes.append("main chain not fully sequential")
        if rollback_ok:
            score += 0.15
            notes.append("both Deploy stages correctly wired to Rollback")
        else:
            notes.append("Rollback not wired from both Deploy stages")
    return min(score, 1.0), notes


def check_branch_01(turns, canvas):
    t = turns[0]
    if not t:
        return 0.0, ["model never called generate_diagram"]
    nodes, arrows = t["nodes"], t["arrows"]
    notes = []
    score = 0.0
    decision = node_by_kw(nodes, "valid")
    if decision and decision.get("shape") == "diamond":
        score += 0.25
        notes.append("decision node correctly shaped as a diamond")
    else:
        notes.append("decision node missing or not a diamond")

    out_edges = [a for a in arrows if decision and a.get("from") == decision["id"]]
    labels = [(a.get("label") or "").strip().lower() for a in out_edges]
    has_yes = any(l in ("yes", "y") for l in labels)
    has_no = any(l in ("no", "n") for l in labels)
    if len(out_edges) >= 2 and has_yes and has_no:
        score += 0.25
        notes.append("two clearly labeled Yes/No branches out of the decision")
    else:
        notes.append(f"decision branches not clearly labeled yes/no: {labels}")

    process = node_by_kw(nodes, "process")
    done = node_by_kw(nodes, "done")
    aset = arrow_set(arrows)
    if process and done and (process["id"], done["id"]) in aset:
        score += 0.25
        notes.append("Process -> Done edge present")
    else:
        notes.append("Process -> Done edge missing")

    retry = node_by_kw(nodes, "retry")
    if retry and decision and (retry["id"], decision["id"]) in aset:
        score += 0.25
        notes.append("Retry -> decision back-edge present (real cycle) and did not break generation")
    else:
        notes.append("Retry -> decision back-edge missing")
    return score, notes


# --- Cases -----------------------------------------------------------------

CASES = [
    {
        "id": "ext-01-splice-node",
        "category": "extend-fidelity",
        "turns": [
            "Create a 3-step pipeline: Ingest, Clean, Store, connected in sequence.",
            "Now add a Validate step between Clean and Store.",
        ],
        "check": check_ext_01,
    },
    {
        "id": "ext-02-reroute-through-cache",
        "category": "extend-fidelity",
        "turns": [
            "Draw an architecture with a Load Balancer connected to two App Servers named App1 and App2, "
            "and both App Servers connect directly to a shared Database.",
            "Add a Redis Cache node positioned between the App Servers and the Database, so both App "
            "Servers go through the cache instead of hitting the database directly.",
        ],
        "check": check_ext_02,
    },
    {
        "id": "grp-01-dashed-worker-pool",
        "category": "group-semantics",
        "turns": [
            "Draw a Task Scheduler that dispatches work to a pool of 4 interchangeable Worker nodes "
            "(no connections between the workers themselves), grouped in a dashed box titled "
            "'Task Workers'. Connect the Scheduler to the Task Workers group as a whole, not to any "
            "single worker.",
        ],
        "check": check_grp_01,
    },
    {
        "id": "grp-02-solid-wired-subsystem",
        "category": "group-semantics",
        "turns": [
            "Draw a 'Resource Manager' subsystem in a solid-bordered box containing a Scheduler wired "
            "to two queues, HighPriorityQueue and LowPriorityQueue (the scheduler feeds both). Outside "
            "the box, connect an API node into the Resource Manager group as a whole.",
        ],
        "check": check_grp_02,
    },
    {
        "id": "grp-03-contradictory-dual-membership",
        "category": "adversarial",
        "turns": [
            "Create two groups: 'Frontend' (dashed) containing a Login node, and 'Backend' (solid) "
            "containing an Auth node. Now also say the Auth node should visually also be considered "
            "part of Frontend.",
        ],
        "check": check_grp_03,
    },
    {
        "id": "adv-01-implied-node-not-dangling",
        "category": "adversarial",
        "turns": [
            "Create a single node called Cache.",
            "Connect Cache to a new step called LegacyMainframe.",
        ],
        "check": check_adv_01,
    },
    {
        "id": "adv-02-invalid-shape-request",
        "category": "adversarial",
        "turns": [
            "Draw a hexagon node called 'Special Step' connected to a rectangle called 'Normal Step'.",
        ],
        "check": check_adv_02,
    },
    {
        "id": "icon-01-architecture-expects-icons",
        "category": "icon-judgment",
        "turns": [
            "Draw a system architecture: a Client sends requests to an API Gateway, which talks to an "
            "Auth Service and a Payments Service, both backed by a shared Database, plus a Message "
            "Queue for async jobs.",
        ],
        "check": check_icon_01,
    },
    {
        "id": "icon-02-trivial-flowchart-no-overkill",
        "category": "icon-judgment",
        "turns": [
            "Draw a plain flowchart for how to make tea: Boil Water, Steep Tea, Add Milk, Serve.",
        ],
        "check": check_icon_02,
    },
    {
        "id": "scale-01-exact-pipeline-count",
        "category": "scale-topology",
        "turns": [
            "Draw a CI/CD pipeline with exactly these 7 stages in sequence: Lint, Unit Test, Build, "
            "Integration Test, Security Scan, Deploy to Staging, Deploy to Production. Also add a "
            "'Rollback' node that both Deploy stages can trigger on failure (an arrow from each Deploy "
            "stage to Rollback).",
        ],
        "check": check_scale_01,
    },
    {
        "id": "branch-01-decision-cycle",
        "category": "scale-topology",
        "turns": [
            "Draw a diamond decision node 'Is Valid?' after an Input node. If yes, go to Process; if "
            "no, go to Reject. Process leads to a final Done node. Reject connects to a Retry node, "
            "and Retry loops back to the Is Valid? decision.",
        ],
        "check": check_branch_01,
    },
]


# --- Runner ----------------------------------------------------------------

def run_eval_suite(case_ids=None, limit=None):
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set (check your .env).")
        return

    cases = CASES
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            print(f"Unknown case id(s), skipping: {sorted(missing)}")
    if limit:
        cases = cases[:limit]
    if not cases:
        print("Nothing to run.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), "evals")
    os.makedirs(out_dir, exist_ok=True)

    print("\n==========================================")
    print(f"  DIAGRAM AGENT EVAL SUITE ({len(cases)} cases, model={MODEL})")
    print("==========================================\n")

    results = []
    start = time.time()

    for idx, case in enumerate(cases, 1):
        canvas = ExcalidrawCanvas(os.path.join(out_dir, f"{case['id']}.excalidraw"))
        messages = []
        turn_args = []

        for prompt in case["turns"]:
            messages.append({"role": "user", "content": prompt})
            turn_args.append(run_turn(messages, canvas))

        score, notes = case["check"](turn_args, canvas)

        group_ids = {g.get("id") for a in turn_args if a for g in (a.get("groups") or [])}
        dangling = dangling_refs(canvas)
        overlaps = overlapping_pairs(canvas, group_ids)

        results.append({
            "id": case["id"], "category": case["category"], "score": score,
            "notes": notes, "dangling": dangling, "overlaps": overlaps,
        })

        with open(os.path.join(out_dir, f"{case['id']}.json"), "w", encoding="utf-8") as f:
            json.dump({"turns": turn_args, "final_elements": canvas.get_elements()}, f, indent=2, ensure_ascii=False)

        status = "PASS" if score >= 0.8 else ("PARTIAL" if score >= 0.4 else "FAIL")
        print(f"[{idx}/{len(cases)}] {case['id']} ({case['category']}): {status}  score={score:.0%}")
        for n in notes:
            print(f"    - {n}")
        if dangling:
            print(f"    ! dangling arrow bindings in final canvas: {dangling}")
        if overlaps:
            print(f"    ! overlapping sibling shapes in final canvas: {overlaps}")
        print()

    duration = time.time() - start

    print("==========================================")
    print("                 SUMMARY                  ")
    print("==========================================")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["score"])
    for cat, scores in by_cat.items():
        print(f"{cat:20s}: {sum(scores) / len(scores):.0%}  (n={len(scores)})")
    overall = sum(r["score"] for r in results) / len(results)
    integrity_fails = sum(1 for r in results if r["dangling"] or r["overlaps"])
    print(f"\nOverall score      : {overall:.0%}")
    print(f"Integrity failures  : {integrity_fails}/{len(results)} cases had dangling refs or overlapping shapes")
    print(f"Total duration      : {duration:.2f}s")
    print(f"Per-case output     : {out_dir}\\<case-id>.json / .excalidraw")
    print("==========================================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagram agent eval suite")
    parser.add_argument("--case", action="append", dest="case_ids", metavar="ID",
                         help="run only this case id (repeatable); default: all cases")
    parser.add_argument("--limit", type=int, default=None,
                         help="run only the first N cases (useful when API quota is tight)")
    args = parser.parse_args()

    run_eval_suite(case_ids=args.case_ids, limit=args.limit)
