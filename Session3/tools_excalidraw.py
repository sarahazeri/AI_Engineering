import json
import os
from typing import Any, Dict, List

class ExcalidrawCanvas:
    """
    Manages the Excalidraw canvas state in RAM and updates the diagram.excalidraw JSON file.
    """
    def __init__(self, filepath="diagram.excalidraw"):
        self.filepath = filepath
        self.elements = []

    def get_elements(self):
        """Returns current elements array."""
        return self.elements

    def clear(self):
        """Clears all elements from the canvas."""
        self.elements = []

    def save(self):
        """Saves current elements to an Excalidraw JSON file."""
        excalidraw_data = {
            "type": "excalidraw",
            "version": 2,
            "source": "http://localhost:8501",
            "elements": self.elements,
            "appState": {
                "gridSize": None,
                "viewBackgroundColor": "#ffffff"
            },
            "files": {}
        }
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(excalidraw_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving Excalidraw file: {e}")
            return False


def _compute_layers(node_ids: List[str], arrows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Assigns each node a column (layer) based on the arrow graph, so nodes are
    positioned by their actual place in the flow instead of raw list order."""
    predecessors = {nid: [] for nid in node_ids}
    for arrow in arrows:
        f, t = arrow.get("from"), arrow.get("to")
        if f in predecessors and t in predecessors and f != t:
            predecessors[t].append(f)

    layer = {nid: 0 for nid in node_ids}
    # Longest-path relaxation (Bellman-Ford style). Bounded iterations keep it
    # safe even if the arrows contain a cycle.
    for _ in range(len(node_ids)):
        changed = False
        for nid in node_ids:
            for pred in predecessors[nid]:
                if layer[pred] + 1 > layer[nid]:
                    layer[nid] = layer[pred] + 1
                    changed = True
        if not changed:
            break
    return layer


def _order_layers(node_ids: List[str], arrows: List[Dict[str, Any]], layer_members: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Reorders nodes within each layer via a barycenter sweep to reduce edge
    crossings: a node is placed near the average row of its already-positioned
    predecessors, so branches (e.g. a decision node's outputs) fan out cleanly
    instead of criss-crossing on their way to a shared downstream node."""
    predecessors = {nid: [] for nid in node_ids}
    for arrow in arrows:
        f, t = arrow.get("from"), arrow.get("to")
        if f in predecessors and t in predecessors and f != t:
            predecessors[t].append(f)

    row_index: Dict[str, float] = {}
    ordered: Dict[int, List[str]] = {}

    for L in sorted(layer_members.keys()):
        members = layer_members[L]
        original_index = {nid: i for i, nid in enumerate(members)}

        def barycenter(nid: str) -> float:
            parent_rows = [row_index[p] for p in predecessors[nid] if p in row_index]
            if parent_rows:
                return sum(parent_rows) / len(parent_rows)
            # No positioned parent (root node, or a node whose parent is in a
            # later layer): keep it in its original relative order.
            return original_index[nid]

        layer_order = sorted(members, key=lambda nid: (barycenter(nid), original_index[nid]))
        ordered[L] = layer_order
        for i, nid in enumerate(layer_order):
            row_index[nid] = i

    return ordered


def _text_width(text: str, font_size: int) -> int:
    """Rough pixel width of a text string at a given font size (Excalidraw's
    default font averages ~0.62 * font_size per character)."""
    return int(len(text) * font_size * 0.62)


def _node_width(label: str, ex_type: str) -> int:
    """Sizes a node from its label length so text isn't clipped or overflowing."""
    base = max(120, _text_width(label, 16) + 40)
    if ex_type == "diamond":
        base = int(base * 1.35)
    return base


def _edge_point(cx: float, cy: float, w: float, h: float, tx: float, ty: float, gap: float = 6):
    """Finds where the line from (cx,cy) to (tx,ty) crosses the boundary of the
    w x h box centered at (cx,cy), pulled back by `gap` pixels. Used so arrows
    touch a node's edge instead of cutting through its center/label."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    hw, hh = w / 2, h / 2
    scale_candidates = []
    if dx != 0:
        scale_candidates.append(hw / abs(dx))
    if dy != 0:
        scale_candidates.append(hh / abs(dy))
    scale = min(scale_candidates)
    ex, ey = cx + dx * scale, cy + dy * scale
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist > 0:
        ex += (dx / dist) * gap
        ey += (dy / dist) * gap
    return ex, ey


def make_diagram_tools(canvas: ExcalidrawCanvas, interactive: bool = True):
    """
    Factory function to create Excalidraw tools bound to a specific canvas instance.
    """

    def generate_diagram(nodes: list = None, arrows: list = None):
        """
        Creates a new diagram from scratch by replacing existing elements.
        """
        canvas.clear()

        nodes = nodes or []
        arrows = arrows or []

        node_ids = [n.get("id", f"node_{i}") for i, n in enumerate(nodes)]
        id_to_node = {nid: nodes[i] for i, nid in enumerate(node_ids)}

        # 1. Lay nodes out by their position in the arrow graph (columns = layers,
        #    branches fan out into rows) instead of a single fixed-spacing row.
        layer = _compute_layers(node_ids, arrows)
        layer_members: Dict[int, List[str]] = {}
        for nid in node_ids:
            layer_members.setdefault(layer[nid], []).append(nid)
        # Reorder each layer by predecessor barycenter to minimize crossings
        # before rows are assigned (important once a node has 2+ branches).
        layer_members = _order_layers(node_ids, arrows, layer_members)

        NODE_HEIGHT = 70
        LAYER_GAP = 100
        BASE_ROW_GAP = 60
        start_x = 100
        start_y = 300

        ex_type_map = {}
        width_map = {}
        for nid in node_ids:
            node = id_to_node[nid]
            n_type = node.get("shape", "rectangle")
            ex_type = "rectangle"
            if n_type == "diamond":
                ex_type = "diamond"
            elif n_type in ["circle", "ellipse"]:
                ex_type = "ellipse"
            ex_type_map[nid] = ex_type
            width_map[nid] = _node_width(node.get("label", ""), ex_type)

        # Arrow labels need room to sit between the two nodes they connect;
        # widen the gap after a layer if its longest outgoing label needs more
        # space than the default LAYER_GAP, so labels never spill into nodes.
        max_out_label_width: Dict[int, int] = {}
        for arrow in arrows:
            a_label = arrow.get("label", "")
            f = arrow.get("from")
            if a_label and f in layer:
                needed = _text_width(a_label, 14) + 40
                L = layer[f]
                max_out_label_width[L] = max(max_out_label_width.get(L, 0), needed)

        # x per layer: cumulative, based on the widest node in each prior layer
        layer_x = {}
        cursor_x = start_x
        for L in sorted(layer_members.keys()):
            layer_x[L] = cursor_x
            widest = max(width_map[nid] for nid in layer_members[L])
            gap = max(LAYER_GAP, max_out_label_width.get(L, 0))
            cursor_x += widest + gap

        # y within a layer: centered around start_y. Layers with more branches
        # get extra breathing room so decision fan-outs don't feel cramped.
        position = {}
        for L, members in layer_members.items():
            count = len(members)
            row_gap = BASE_ROW_GAP + (20 * max(0, count - 2))
            for i, nid in enumerate(members):
                y = start_y + (i - (count - 1) / 2) * (NODE_HEIGHT + row_gap)
                position[nid] = (layer_x[L], y)

        # 2. Build shape + bound text elements
        shape_lookup: Dict[str, Dict[str, Any]] = {}
        for nid in node_ids:
            node = id_to_node[nid]
            label = node.get("label", "")
            stroke_color = node.get("stroke_color", "#1e1e1e")
            bg_color = node.get("bg_color", "#e0e7ff")
            ex_type = ex_type_map[nid]
            width = width_map[nid]
            height = NODE_HEIGHT
            pos_x, pos_y = position[nid]

            text_id = f"text_{nid}"
            shape_element = {
                "id": nid,
                "type": ex_type,
                "x": pos_x,
                "y": pos_y,
                "width": width,
                "height": height,
                "angle": 0,
                "strokeColor": stroke_color,
                "backgroundColor": bg_color,
                "fillStyle": "solid",
                "strokeWidth": 2,
                "strokeStyle": "solid",
                "roughness": 1,
                "opacity": 100,
                "groupIds": [],
                "frameId": None,
                "roundness": {"type": 3},
                # Two-way binding: the shape must list its own label so Excalidraw
                # treats it as a real bound container instead of overlapping text.
                "boundElements": [{"id": text_id, "type": "text"}] if label else [],
                "isDeleted": False
            }
            canvas.elements.append(shape_element)
            shape_lookup[nid] = {
                "element": shape_element,
                "x": pos_x, "y": pos_y, "width": width, "height": height
            }

            if label:
                text_element = {
                    "id": text_id,
                    "type": "text",
                    "x": pos_x + 15,
                    "y": pos_y + (height / 2) - 10,
                    "width": width - 30,
                    "height": 20,
                    "angle": 0,
                    "strokeColor": "#1e1e1e",
                    "backgroundColor": "transparent",
                    "fillStyle": "hachure",
                    "strokeWidth": 1,
                    "strokeStyle": "solid",
                    "roughness": 0,
                    "opacity": 100,
                    "text": label,
                    "fontSize": 16,
                    "fontFamily": 1,
                    "textAlign": "center",
                    "verticalAlign": "middle",
                    "isDeleted": False,
                    "containerId": nid
                }
                canvas.elements.append(text_element)

        # 3. Build arrows bound to the shapes they connect, anchored at the
        #    actual edge point between the two node centers (not a fixed
        #    mid-height line), so they no longer cut through unrelated nodes.
        for idx, arrow in enumerate(arrows):
            from_id = arrow.get("from")
            to_id = arrow.get("to")
            a_label = arrow.get("label", "")

            if from_id in shape_lookup and to_id in shape_lookup:
                src = shape_lookup[from_id]
                dst = shape_lookup[to_id]
                src_cx, src_cy = src["x"] + src["width"] / 2, src["y"] + src["height"] / 2
                dst_cx, dst_cy = dst["x"] + dst["width"] / 2, dst["y"] + dst["height"] / 2

                sx, sy = _edge_point(src_cx, src_cy, src["width"], src["height"], dst_cx, dst_cy)
                ex, ey = _edge_point(dst_cx, dst_cy, dst["width"], dst["height"], src_cx, src_cy)

                dx, dy = ex - sx, ey - sy
                arrow_id = f"arrow_{idx}"

                arrow_element = {
                    "id": arrow_id,
                    "type": "arrow",
                    "x": sx,
                    "y": sy,
                    "width": dx,
                    "height": dy,
                    "angle": 0,
                    "strokeColor": "#1e1e1e",
                    "backgroundColor": "transparent",
                    "fillStyle": "hachure",
                    "strokeWidth": 2,
                    "strokeStyle": "solid",
                    "roughness": 1,
                    "opacity": 100,
                    "points": [[0, 0], [dx, dy]],
                    "startBinding": {"elementId": from_id, "focus": 0, "gap": 6},
                    "endBinding": {"elementId": to_id, "focus": 0, "gap": 6},
                    "endArrowhead": "arrow",
                    "startArrowhead": None,
                    "isDeleted": False
                }
                canvas.elements.append(arrow_element)
                src["element"]["boundElements"].append({"id": arrow_id, "type": "arrow"})
                dst["element"]["boundElements"].append({"id": arrow_id, "type": "arrow"})

                if a_label:
                    mx, my = sx + dx / 2, sy + dy / 2
                    label_w = _text_width(a_label, 14) + 10
                    # Float the label clear above the line instead of sitting
                    # on top of it, so it doesn't blend into the arrow/nodes.
                    canvas.elements.append({
                        "id": f"text_{arrow_id}",
                        "type": "text",
                        "x": mx - label_w / 2,
                        "y": my - 28,
                        "width": label_w,
                        "height": 20,
                        "angle": 0,
                        "strokeColor": "#1e1e1e",
                        "backgroundColor": "#ffffff",
                        "fillStyle": "hachure",
                        "strokeWidth": 1,
                        "strokeStyle": "solid",
                        "roughness": 0,
                        "opacity": 100,
                        "text": a_label,
                        "fontSize": 14,
                        "fontFamily": 1,
                        "textAlign": "center",
                        "verticalAlign": "middle",
                        "isDeleted": False
                    })

        canvas.save()
        return f"Successfully generated Excalidraw diagram with {len(canvas.elements)} elements in '{canvas.filepath}'."

    # Tool JSON Schemas for Llama/Groq API
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "generate_diagram",
                "description": "Generates a complete Excalidraw diagram from nodes and connecting arrows. Layout (position, spacing, edge routing) is computed automatically from the arrow graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "description": "List of diagram shapes/nodes",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique node identifier"},
                                    "label": {"type": "string", "description": "Text inside the node"},
                                    "shape": {"type": "string", "enum": ["rectangle", "diamond", "ellipse"], "description": "Node shape"},
                                    "bg_color": {"type": "string", "description": "Hex background color e.g. #e0e7ff"}
                                },
                                "required": ["id", "label"]
                            }
                        },
                        "arrows": {
                            "type": "array",
                            "description": "List of connecting arrows between nodes",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "description": "Source node ID"},
                                    "to": {"type": "string", "description": "Target node ID"},
                                    "label": {"type": "string", "description": "Optional label on arrow"}
                                },
                                "required": ["from", "to"]
                            }
                        }
                    },
                    "required": ["nodes", "arrows"]
                }
            }
        }
    ]

    tool_map = {
        "generate_diagram": generate_diagram
    }

    return tool_map, tools_schema
