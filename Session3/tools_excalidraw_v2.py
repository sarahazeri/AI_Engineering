"""
Extends tools_excalidraw.py with two new capabilities needed for
architecture-style diagrams (e.g. a "Resource Manager" box containing a
stack of queues, next to a "Task Workers" box containing a grid of
worker icons):

1. Groups - nodes can be placed inside a titled container (solid or
   dashed border). A group's children are laid out internally (stacked
   by their own local relationships, or gridded if they have none) and
   the group is then treated as a single block in the outer diagram.
2. Icons - an optional emoji/glyph rendered above a node's label.

The underlying per-node primitives (text sizing, node sizing, and the
edge-boundary math) are reused as-is from tools_excalidraw.py.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

from tools_excalidraw import ExcalidrawCanvas, _text_width, _node_width, _edge_point


def _node_height(has_icon: bool) -> int:
    return 92 if has_icon else 70


def _compute_layers(item_ids: List[str], edges: List[Tuple[str, str]]) -> Dict[str, int]:
    """Same longest-path layering as tools_excalidraw._compute_layers, but
    generic over any item id (a plain node, or a whole group treated as
    one block) and a plain (from, to) edge list."""
    predecessors = {iid: [] for iid in item_ids}
    for f, t in edges:
        if f in predecessors and t in predecessors and f != t:
            predecessors[t].append(f)

    layer = {iid: 0 for iid in item_ids}
    for _ in range(len(item_ids)):
        changed = False
        for iid in item_ids:
            for pred in predecessors[iid]:
                if layer[pred] + 1 > layer[iid]:
                    layer[iid] = layer[pred] + 1
                    changed = True
        if not changed:
            break
    return layer


def _order_layers(item_ids: List[str], edges: List[Tuple[str, str]], layer_members: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Same barycenter crossing-reduction sweep as tools_excalidraw._order_layers,
    generic over item id / edge list."""
    predecessors = {iid: [] for iid in item_ids}
    for f, t in edges:
        if f in predecessors and t in predecessors and f != t:
            predecessors[t].append(f)

    row_index: Dict[str, float] = {}
    ordered: Dict[int, List[str]] = {}

    for L in sorted(layer_members.keys()):
        members = layer_members[L]
        original_index = {iid: i for i, iid in enumerate(members)}

        def barycenter(iid: str) -> float:
            parent_rows = [row_index[p] for p in predecessors[iid] if p in row_index]
            if parent_rows:
                return sum(parent_rows) / len(parent_rows)
            return original_index[iid]

        layer_order = sorted(members, key=lambda iid: (barycenter(iid), original_index[iid]))
        ordered[L] = layer_order
        for i, iid in enumerate(layer_order):
            row_index[iid] = i

    return ordered


def _layout_items(item_ids: List[str], sizes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]],
                   edge_labels: Optional[Dict[Tuple[str, str], str]] = None):
    """Lays `item_ids` out into columns (by graph depth) and centered rows
    within a column (via the barycenter ordering), same approach as
    tools_excalidraw.generate_diagram but generalized to arbitrary item
    sizes (a group block is just a very large "node" here) and returning
    local, un-anchored CENTER coordinates plus the layer map (so the
    caller can detect skip-layer edges and route them with a bend).
    Returns (positions: {id: (cx, cy)}, layer: {id: int})."""
    edge_labels = edge_labels or {}
    layer = _compute_layers(item_ids, edges)
    layer_members: Dict[int, List[str]] = {}
    for iid in item_ids:
        layer_members.setdefault(layer[iid], []).append(iid)
    layer_members = _order_layers(item_ids, edges, layer_members)

    LAYER_GAP = 100
    BASE_ROW_GAP = 60

    # Widen the gap after a layer if its longest outgoing edge label needs
    # more horizontal room than the default gap provides.
    max_out_label_width: Dict[int, int] = {}
    for (f, t), lbl in edge_labels.items():
        if lbl and f in layer:
            needed = _text_width(lbl, 14) + 40
            L = layer[f]
            max_out_label_width[L] = max(max_out_label_width.get(L, 0), needed)

    layer_x: Dict[int, float] = {}
    cursor_x = 0.0
    for L in sorted(layer_members.keys()):
        layer_x[L] = cursor_x
        widest = max(sizes[iid][0] for iid in layer_members[L])
        gap = max(LAYER_GAP, max_out_label_width.get(L, 0))
        cursor_x += widest + gap

    positions: Dict[str, Tuple[float, float]] = {}
    for L, members in layer_members.items():
        row_gap = BASE_ROW_GAP + (20 * max(0, len(members) - 2))
        total_h = sum(sizes[iid][1] for iid in members) + row_gap * (len(members) - 1)
        cursor_y = -total_h / 2
        for iid in members:
            w, h = sizes[iid]
            cx = layer_x[L] + w / 2
            cy = cursor_y + h / 2
            positions[iid] = (cx, cy)
            cursor_y += h + row_gap

    return positions, layer


def _grid_layout(item_ids: List[str], sizes: Dict[str, Tuple[float, float]]):
    """Arranges items with no relationships to each other in a roughly
    square grid (used for icon galleries like a pool of worker nodes)."""
    n = len(item_ids)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    COL_GAP, ROW_GAP = 40, 30

    col_widths = [0.0] * cols
    row_heights = [0.0] * rows
    cell = {}
    for idx, iid in enumerate(item_ids):
        r, c = divmod(idx, cols)
        cell[iid] = (r, c)
        w, h = sizes[iid]
        col_widths[c] = max(col_widths[c], w)
        row_heights[r] = max(row_heights[r], h)

    col_x = [0.0] * cols
    for c in range(1, cols):
        col_x[c] = col_x[c - 1] + col_widths[c - 1] + COL_GAP
    row_y = [0.0] * rows
    for r in range(1, rows):
        row_y[r] = row_y[r - 1] + row_heights[r - 1] + ROW_GAP

    positions = {}
    for iid in item_ids:
        r, c = cell[iid]
        w, h = sizes[iid]
        cx = col_x[c] + col_widths[c] / 2
        cy = row_y[r] + row_heights[r] / 2
        positions[iid] = (cx, cy)
    return positions


def _route_points(src_cx, src_cy, src_w, src_h, dst_cx, dst_cy, dst_w, dst_h,
                   skip: bool, lane_y: float, gap: float = 6):
    """Returns the arrow's point list (relative to its own origin) plus its
    absolute origin. For adjacent items this is a straight edge-to-edge
    line. For a "skip" edge (it jumps over at least one column that sits
    physically between the two endpoints) it instead exits upward to a
    shared lane above the whole diagram, travels across, then drops back
    down - so it never cuts through an unrelated node in between."""
    if not skip:
        sx, sy = _edge_point(src_cx, src_cy, src_w, src_h, dst_cx, dst_cy, gap)
        ex, ey = _edge_point(dst_cx, dst_cy, dst_w, dst_h, src_cx, src_cy, gap)
        return (sx, sy), [(0, 0), (ex - sx, ey - sy)]

    p0x, p0y = _edge_point(src_cx, src_cy, src_w, src_h, src_cx, lane_y, gap)
    p3x, p3y = _edge_point(dst_cx, dst_cy, dst_w, dst_h, dst_cx, lane_y, gap)
    p1 = (p0x, lane_y)
    p2 = (p3x, lane_y)
    origin = (p0x, p0y)
    points = [
        (0, 0),
        (p1[0] - p0x, p1[1] - p0y),
        (p2[0] - p0x, p2[1] - p0y),
        (p3x - p0x, p3y - p0y),
    ]
    return origin, points


def make_diagram_tools(canvas: ExcalidrawCanvas, interactive: bool = True):
    """
    Factory function to create Excalidraw tools bound to a specific canvas
    instance. Adds `groups` support (titled containers) and per-node
    `icon` glyphs on top of the plain node/arrow tool.
    """

    def generate_diagram(nodes: list = None, arrows: list = None, groups: list = None):
        """Creates a new diagram from scratch by replacing existing elements."""
        canvas.clear()

        nodes = nodes or []
        arrows = arrows or []
        groups = groups or []

        node_ids = [n.get("id", f"node_{i}") for i, n in enumerate(nodes)]
        id_to_node = {nid: nodes[i] for i, nid in enumerate(node_ids)}
        group_meta = {g.get("id"): g for g in groups if g.get("id")}

        GROUP_PADDING = 30
        TITLE_HEIGHT = 36

        # 1. Partition nodes into their group (if any) or "ungrouped"
        group_children: Dict[str, List[str]] = {}
        ungrouped: List[str] = []
        for nid in node_ids:
            gid = id_to_node[nid].get("group")
            if gid and gid in group_meta:
                group_children.setdefault(gid, []).append(nid)
            else:
                ungrouped.append(nid)

        # 2. Per-node visual attributes (shape type, size, icon)
        ex_type_map: Dict[str, str] = {}
        size_map: Dict[str, Tuple[float, float]] = {}
        display_text: Dict[str, str] = {}
        for nid in node_ids:
            node = id_to_node[nid]
            n_type = node.get("shape", "rectangle")
            ex_type = "rectangle"
            if n_type == "diamond":
                ex_type = "diamond"
            elif n_type in ("circle", "ellipse"):
                ex_type = "ellipse"
            ex_type_map[nid] = ex_type

            label = node.get("label", "")
            icon = node.get("icon", "")
            display_text[nid] = f"{icon}\n{label}" if icon else label
            width = _node_width(label, ex_type)
            height = _node_height(bool(icon))
            size_map[nid] = (width, height)

        # 3. Lay out each group's children internally: by their own local
        #    arrows if any exist (e.g. a scheduler wired to its queues), or
        #    as a grid if the children have no relationships to each other
        #    (e.g. a pool of interchangeable worker icons).
        group_layout: Dict[str, Dict[str, Any]] = {}
        for gid, children in group_children.items():
            internal_arrows = [a for a in arrows if a.get("from") in children and a.get("to") in children]
            edges = [(a["from"], a["to"]) for a in internal_arrows]
            edge_labels = {(a["from"], a["to"]): a.get("label", "") for a in internal_arrows}
            sizes = {cid: size_map[cid] for cid in children}

            if edges:
                local_pos, _ = _layout_items(children, sizes, edges, edge_labels)
            else:
                local_pos = _grid_layout(children, sizes)

            xs = [local_pos[c][0] - sizes[c][0] / 2 for c in children]
            xe = [local_pos[c][0] + sizes[c][0] / 2 for c in children]
            ys = [local_pos[c][1] - sizes[c][1] / 2 for c in children]
            ye = [local_pos[c][1] + sizes[c][1] / 2 for c in children]
            min_x, min_y = min(xs), min(ys)
            content_w, content_h = max(xe) - min_x, max(ye) - min_y

            title = group_meta[gid].get("title", gid)
            group_w = max(content_w + GROUP_PADDING * 2, _text_width(title, 18) + GROUP_PADDING * 2)
            group_h = content_h + GROUP_PADDING * 2 + TITLE_HEIGHT

            shifted = {
                c: (local_pos[c][0] - min_x + GROUP_PADDING, local_pos[c][1] - min_y + GROUP_PADDING + TITLE_HEIGHT)
                for c in children
            }

            group_layout[gid] = {
                "local_pos": shifted,
                "width": group_w,
                "height": group_h,
                "internal_arrows": internal_arrows,
            }

        def macro_of(nid: str) -> str:
            # An arrow may target a whole group (e.g. "run task" pointing at
            # the Task Workers container itself) rather than one of its nodes.
            if nid in group_layout:
                return nid
            node = id_to_node.get(nid)
            gid = node.get("group") if node else None
            return gid if gid in group_layout else nid

        # 4. Outer diagram: groups + ungrouped nodes, laid out the same way
        macro_ids = list(group_layout.keys()) + ungrouped
        macro_sizes: Dict[str, Tuple[float, float]] = {}
        for gid, gl in group_layout.items():
            macro_sizes[gid] = (gl["width"], gl["height"])
        for nid in ungrouped:
            macro_sizes[nid] = size_map[nid]

        macro_edges: List[Tuple[str, str]] = []
        macro_edge_labels: Dict[Tuple[str, str], str] = {}
        seen = set()
        for a in arrows:
            f, t = a.get("from"), a.get("to")
            if (f not in id_to_node and f not in group_layout) or (t not in id_to_node and t not in group_layout):
                continue
            mf, mt = macro_of(f), macro_of(t)
            if mf != mt and (mf, mt) not in seen:
                seen.add((mf, mt))
                macro_edges.append((mf, mt))
                macro_edge_labels[(mf, mt)] = a.get("label", "")

        macro_pos, macro_layer = _layout_items(macro_ids, macro_sizes, macro_edges, macro_edge_labels)

        # 5. Resolve every node's absolute (center) position: ungrouped
        #    nodes sit directly at their macro position, grouped nodes are
        #    the group's local layout shifted to the group's macro position.
        abs_pos: Dict[str, Tuple[float, float, float, float]] = {}
        for nid in ungrouped:
            cx, cy = macro_pos[nid]
            w, h = macro_sizes[nid]
            abs_pos[nid] = (cx, cy, w, h)

        for gid, gl in group_layout.items():
            gcx, gcy = macro_pos[gid]
            gw, gh = gl["width"], gl["height"]
            gx0, gy0 = gcx - gw / 2, gcy - gh / 2
            gl["abs_x0"], gl["abs_y0"] = gx0, gy0
            for cid, (lx, ly) in gl["local_pos"].items():
                w, h = size_map[cid]
                abs_pos[cid] = (gx0 + lx, gy0 + ly, w, h)

        # 6. Emit group containers first (so nodes render on top of them)
        shape_lookup: Dict[str, Dict[str, Any]] = {}
        for gid, gl in group_layout.items():
            gx0, gy0 = gl["abs_x0"], gl["abs_y0"]
            gw, gh = gl["width"], gl["height"]
            style = group_meta[gid].get("style", "solid")
            container = {
                "id": gid, "type": "rectangle", "x": gx0, "y": gy0,
                "width": gw, "height": gh, "angle": 0,
                "strokeColor": "#475569", "backgroundColor": "transparent", "fillStyle": "solid",
                "strokeWidth": 2, "strokeStyle": "dashed" if style == "dashed" else "solid",
                "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
                "roundness": {"type": 3}, "boundElements": [], "isDeleted": False
            }
            canvas.elements.append(container)
            shape_lookup[gid] = {"element": container, "x": gx0, "y": gy0, "width": gw, "height": gh,
                                  "cx": gx0 + gw / 2, "cy": gy0 + gh / 2}

            title = group_meta[gid].get("title", gid)
            canvas.elements.append({
                "id": f"title_{gid}", "type": "text", "x": gx0 + 14, "y": gy0 + 8,
                "width": gw - 28, "height": 22, "angle": 0,
                "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "hachure",
                "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
                "text": title, "fontSize": 18, "fontFamily": 1, "textAlign": "center",
                "verticalAlign": "top", "isDeleted": False
            })

        # 7. Emit node shapes + bound labels
        for nid in node_ids:
            node = id_to_node[nid]
            cx, cy, w, h = abs_pos[nid]
            x, y = cx - w / 2, cy - h / 2
            ex_type = ex_type_map[nid]
            stroke_color = node.get("stroke_color", "#1e1e1e")
            bg_color = node.get("bg_color", "#e0e7ff")
            text_id = f"text_{nid}"

            shape_element = {
                "id": nid, "type": ex_type, "x": x, "y": y, "width": w, "height": h, "angle": 0,
                "strokeColor": stroke_color, "backgroundColor": bg_color, "fillStyle": "solid",
                "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
                "groupIds": [], "frameId": None, "roundness": {"type": 3},
                "boundElements": [{"id": text_id, "type": "text"}] if display_text[nid] else [],
                "isDeleted": False
            }
            canvas.elements.append(shape_element)
            shape_lookup[nid] = {"element": shape_element, "x": x, "y": y, "width": w, "height": h, "cx": cx, "cy": cy}

            if display_text[nid]:
                canvas.elements.append({
                    "id": text_id, "type": "text", "x": x + 10, "y": y + 6,
                    "width": w - 20, "height": h - 12, "angle": 0,
                    "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "hachure",
                    "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
                    "text": display_text[nid], "fontSize": 16, "fontFamily": 1, "textAlign": "center",
                    "verticalAlign": "middle", "isDeleted": False, "containerId": nid
                })

        # 8. Diagram-wide lane for elbow-routed (skip-layer) arrows: just
        #    above the topmost element, shared by every such arrow.
        all_tops = [v["y"] for v in shape_lookup.values()]
        lane_y = (min(all_tops) - 40) if all_tops else 0

        def emit_arrow(arrow_id: str, from_id: str, to_id: str, a_label: str, skip: bool):
            src, dst = shape_lookup[from_id], shape_lookup[to_id]
            origin, points = _route_points(
                src["cx"], src["cy"], src["width"], src["height"],
                dst["cx"], dst["cy"], dst["width"], dst["height"],
                skip, lane_y
            )
            ox, oy = origin
            end_dx, end_dy = points[-1]
            arrow_element = {
                "id": arrow_id, "type": "arrow", "x": ox, "y": oy,
                "width": end_dx, "height": end_dy, "angle": 0,
                "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "hachure",
                "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
                "points": [list(p) for p in points],
                "startBinding": {"elementId": from_id, "focus": 0, "gap": 6},
                "endBinding": {"elementId": to_id, "focus": 0, "gap": 6},
                "endArrowhead": "arrow", "startArrowhead": None, "isDeleted": False
            }
            canvas.elements.append(arrow_element)
            src["element"]["boundElements"].append({"id": arrow_id, "type": "arrow"})
            dst["element"]["boundElements"].append({"id": arrow_id, "type": "arrow"})

            if a_label:
                # Put the label on the flattest (usually horizontal) segment,
                # floated clear above the line instead of sitting on top of it.
                seg = max(range(len(points) - 1), key=lambda i: abs(points[i + 1][0] - points[i][0]))
                mx = ox + (points[seg][0] + points[seg + 1][0]) / 2
                my = oy + (points[seg][1] + points[seg + 1][1]) / 2
                label_w = _text_width(a_label, 14) + 10
                canvas.elements.append({
                    "id": f"text_{arrow_id}", "type": "text",
                    "x": mx - label_w / 2, "y": my - 28, "width": label_w, "height": 20, "angle": 0,
                    "strokeColor": "#1e1e1e", "backgroundColor": "#ffffff", "fillStyle": "hachure",
                    "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0, "opacity": 100,
                    "text": a_label, "fontSize": 14, "fontFamily": 1, "textAlign": "center",
                    "verticalAlign": "middle", "isDeleted": False
                })

        # 9. Emit each group's internal arrows (local layer, so skip-check
        #    uses that group's own layering, not the outer diagram's)
        arrow_idx = 0
        for gid, children in group_children.items():
            internal_arrows = group_layout[gid]["internal_arrows"]
            if not internal_arrows:
                continue
            edges = [(a["from"], a["to"]) for a in internal_arrows]
            local_layer = _compute_layers(children, edges)
            for a in internal_arrows:
                f, t = a["from"], a["to"]
                skip = abs(local_layer.get(t, 0) - local_layer.get(f, 0)) > 1
                emit_arrow(f"arrow_{arrow_idx}", f, t, a.get("label", ""), skip)
                arrow_idx += 1

        # 10. Emit every remaining arrow (cross-group or between ungrouped
        #     nodes), using the outer diagram's macro layer for skip-check.
        for a in arrows:
            f, t = a.get("from"), a.get("to")
            if f not in shape_lookup or t not in shape_lookup:
                continue
            if macro_of(f) == macro_of(t) and macro_of(f) in group_layout:
                continue  # already emitted as an internal group arrow above
            mf, mt = macro_of(f), macro_of(t)
            skip = abs(macro_layer.get(mt, 0) - macro_layer.get(mf, 0)) > 1
            emit_arrow(f"arrow_{arrow_idx}", f, t, a.get("label", ""), skip)
            arrow_idx += 1

        canvas.save()
        return f"Successfully generated Excalidraw diagram with {len(canvas.elements)} elements in '{canvas.filepath}'."

    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "generate_diagram",
                "description": (
                    "Generates a complete Excalidraw diagram from nodes, arrows, and optional groups. "
                    "Layout (position, spacing, edge routing) is computed automatically. Use `groups` to "
                    "draw a titled container box around a set of related nodes (e.g. a subsystem that "
                    "contains several internal components), like a 'Resource Manager' box containing "
                    "several queues, or a dashed 'Task Workers' box containing a pool of worker icons."
                ),
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
                                    "bg_color": {"type": "string", "description": "Hex background color e.g. #e0e7ff"},
                                    "icon": {"type": "string", "description": "Optional single emoji glyph shown above the label, e.g. an icon summarizing the node (server, mail/queue, gear, database, etc.)"},
                                    "group": {"type": "string", "description": "Optional id of a group (from the `groups` list) this node belongs to. Nodes sharing a group are drawn inside the same titled container box."}
                                },
                                "required": ["id", "label"]
                            }
                        },
                        "arrows": {
                            "type": "array",
                            "description": "List of connecting arrows between nodes (nodes may be in different groups, the same group, or ungrouped)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string", "description": "Source node ID"},
                                    "to": {"type": "string", "description": "Target node ID"},
                                    "label": {"type": "string", "description": "Optional label on arrow"}
                                },
                                "required": ["from", "to"]
                            }
                        },
                        "groups": {
                            "type": "array",
                            "description": "Optional list of titled container boxes. Reference a group's id from a node's `group` field to place that node inside it.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique group identifier"},
                                    "title": {"type": "string", "description": "Title shown at the top of the container box"},
                                    "style": {"type": "string", "enum": ["solid", "dashed"], "description": "Border style of the container. Use 'dashed' for a looser/optional grouping (e.g. a pool of interchangeable workers), 'solid' for a concrete subsystem boundary."}
                                },
                                "required": ["id", "title"]
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
