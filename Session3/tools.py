import json
import xml.sax.saxutils as xml_escape
from typing import List, Dict, Any

class Canvas:
    """Manage the state of diagram elements and smart rendering of the SVG file"""
    def __init__(self, output_file: str = "canvas.svg"):
        self.elements: List[Dict[str, Any]] = []
        self.output_file = output_file

    def set_elements(self, elements: List[Dict[str, Any]], auto_save: bool = False):
        """Replace the canvas elements with new elements"""
        self.elements = elements
        if auto_save:
            self.save_svg()

    def add_elements(self, new_elements: List[Dict[str, Any]], auto_save: bool = False):
        """Add new elements to the previous canvas elements (Additive)"""
        self.elements.extend(new_elements)
        if auto_save:
            self.save_svg()

    def update_element(self, element_id: str, updates: Dict[str, Any], auto_save: bool = False) -> bool:
        """Update the properties of a specific element with a specific ID"""
        for el in self.elements:
            if el.get("id") == element_id:
                el.update(updates)
                if auto_save:
                    self.save_svg()
                return True
        return False

    def get_elements(self) -> List[Dict[str, Any]]:
        """Get the list of all current canvas elements"""
        return self.elements

    def clear(self):
        """Clear all canvas elements"""
        self.elements = []

    def save_svg(self) -> str:
        """Render the current elements and save in the SVG file"""
        svg_code = self._generate_svg()
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(svg_code)
            return f"Diagram successfully saved in the file '{self.output_file}'."
        except Exception as e:
            return f"Error in saving the SVG file: {e}"

    def _generate_svg(self) -> str:
        """Generate the SVG code string based on the canvas elements"""
        width = 1000
        height = 600
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '  <defs>',
            '    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">',
            '      <feDropShadow dx="2" dy="3" stdDeviation="3" flood-opacity="0.12" />',
            '    </filter>',
            '    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '      <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />',
            '    </marker>',
            '  </defs>',
            '  <style>',
            '    .node-rect { stroke: #334155; stroke-width: 2; rx: 8; filter: url(#shadow); }',
            '    .node-diamond { stroke: #334155; stroke-width: 2; filter: url(#shadow); }',
            '    .node-circle { stroke: #334155; stroke-width: 2; filter: url(#shadow); }',
            '    .text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; font-weight: 500; text-anchor: middle; dominant-baseline: central; fill: #0f172a; }',
            '    .arrow-text { font-family: sans-serif; font-size: 11px; fill: #64748b; font-weight: 600; text-anchor: middle; }',
            '    .arrow { stroke: #475569; stroke-width: 2; marker-end: url(#arrowhead); fill: none; }',
            '  </style>',
            '  <rect width="100%" height="100%" fill="#f8fafc" />'
        ]

        nodes = [el for el in self.elements if el.get("type") != "arrow"]
        arrows = [el for el in self.elements if el.get("type") == "arrow"]
        
        node_info = {}
        start_x = 120
        start_y = 200
        spacing_x = 230

        # Calculate the dimensions and coordinates of each node
        for idx, node in enumerate(nodes):
            nid = node.get("id")
            label = xml_escape.escape(str(node.get("label", node.get("text", nid))))
            ntype = node.get("type", "rectangle").lower()
            
            calculated_width = max(130, len(label) * 9 + 24)
            node_h = 60
            
            # Support for direct x and y if present, or smart handling of error branches
            if "x" in node and "y" in node:
                x = node["x"]
                y = node["y"]
            else:
                # If the element is related to an error or failure, place it at the bottom row
                if "fail" in nid.lower() or "error" in nid.lower() or "invalid" in label.lower():
                    x = start_x + (1.5 * spacing_x)
                    y = start_y + 160
                else:
                    x = start_x + idx * spacing_x
                    y = start_y

            node_info[nid] = {
                "x": x,
                "y": y,
                "width": calculated_width,
                "height": node_h,
                "type": ntype,
                "label": label,
                "color": node.get("color", node.get("backgroundColor", None))
            }

        # Draw the nodes
        for nid, info in node_info.items():
            cx, cy = info["x"], info["y"]
            w, h = info["width"], info["height"]
            label = info["label"]
            ntype = info["type"]
            
            default_colors = {
                "diamond": "#fef3c7",
                "circle": "#dcfce7",
                "rectangle": "#e2e8f0"
            }
            
            # Automatically change the color of elements related to errors to a light red
            if "fail" in nid.lower() or "error" in nid.lower() or "invalid" in label.lower():
                fill_color = info["color"] if info["color"] else "#fee2e2"
            else:
                fill_color = info["color"] if info["color"] else default_colors.get(ntype, "#e2e8f0")

            if ntype == "diamond":
                hw, hh = w / 1.1, h / 1.1
                points = f"{cx},{cy-hh} {cx+hw},{cy} {cx},{cy+hh} {cx-hw},{cy}"
                svg_parts.append(f'  <polygon points="{points}" class="node-diamond" fill="{fill_color}" />')
            elif ntype == "circle":
                r = max(w, h) / 2
                svg_parts.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" class="node-circle" fill="{fill_color}" />')
            else:  # Rectangle
                rx = cx - (w / 2)
                ry = cy - (h / 2)
                svg_parts.append(f'  <rect x="{rx}" y="{ry}" width="{w}" height="{h}" class="node-rect" fill="{fill_color}" />')

            svg_parts.append(f'  <text x="{cx}" y="{cy}" class="text">{label}</text>')

        # Draw the arrows with vertical and horizontal connections calculations
        for arrow in arrows:
            from_id = arrow.get("from")
            to_id = arrow.get("to")
            label = xml_escape.escape(str(arrow.get("label", arrow.get("text", ""))))
            
            if from_id in node_info and to_id in node_info:
                src = node_info[from_id]
                dst = node_info[to_id]
                
                # Vertical connection (if the destination element is in the bottom row)
                if abs(dst["y"] - src["y"]) > 80:
                    fx = src["x"]
                    fy = src["y"] + (src["height"] / (1.1 if src["type"] == "diamond" else 2))
                    tx = dst["x"]
                    ty = dst["y"] - (dst["height"] / (1.1 if dst["type"] == "diamond" else 2))
                else:
                    # Standard horizontal connection
                    fx = src["x"] + (src["width"] / (1.1 if src["type"] == "diamond" else 2))
                    fy = src["y"]
                    tx = dst["x"] - (dst["width"] / (1.1 if dst["type"] == "diamond" else 2))
                    ty = dst["y"]

                svg_parts.append(f'  <line x1="{fx}" y1="{fy}" x2="{tx}" y2="{ty}" class="arrow" />')
                
                # Add labels on the arrows (like Yes / No)
                if label:
                    mx, my = (fx + tx) / 2, (fy + ty) / 2 - 8
                    svg_parts.append(f'  <text x="{mx}" y="{my}" class="arrow-text">{label}</text>')

        svg_parts.append('</svg>')
        return "\n".join(svg_parts)


def make_diagram_tools(canvas: Canvas, interactive: bool = True):    
    
    if interactive:
        """Factory for creating tools connected to the canvas"""
    else:
        """Factory for creating tools connected to the canvas without interactive mode"""

    
    def generate_diagram(elements: List[Dict[str, Any]]) -> str:
        """Draw a complete diagram and request confirmation before saving to file"""
        # try:
        #     confirm = input(f"\n⚠️ Do you want to save the new diagram in the file '{canvas.output_file}'? (y/n): ").strip().lower()
            
        #     if confirm == 'y':
        canvas.set_elements(elements)
        save_status = canvas.save_svg()
        return f"Successfully generated diagram with {len(elements)} elements. {save_status}"
            # else:
            #     return "User cancelled saving the diagram to file. No changes were saved to canvas.svg."
        # except Exception as e:
        #     return f"Error in generate_diagram: {str(e)}"

    def add_elements(elements: List[Dict[str, Any]]) -> str:
        """Add new elements to the current canvas without deleting the previous elements"""
        try:
            confirm = input(f"\n⚠️ Do you want to add the new elements in the file '{canvas.output_file}'? (y/n): ").strip().lower()
            
            if confirm == 'y':
                canvas.add_elements(elements)
                save_status = canvas.save_svg()
                return f"Successfully added {len(elements)} new elements to canvas. {save_status}"
            else:
                return "User cancelled adding elements. File canvas.svg left untouched."
        except Exception as e:
            return f"Error in add_elements: {str(e)}"

    def modify_diagram(id: str, updates: Dict[str, Any]) -> str:
        """Modify the properties of a specific element and request confirmation before updating the file"""
        try:
            confirm = input(f"\n⚠️ Do you want to save the changes of the element '{id}' in the file '{canvas.output_file}'? (y/n): ").strip().lower()
            
            if confirm == 'y':
                success = canvas.update_element(id, updates)
                if success:
                    save_status = canvas.save_svg()
                    return f"Successfully updated element '{id}'. {save_status}"
                else:
                    return f"Error: Element with id '{id}' not found on canvas."
            else:
                return "User cancelled modifying the diagram. File canvas.svg left untouched."
        except Exception as e:
            return f"Error in modify_diagram: {str(e)}"

    tool_map = {
        "generate_diagram": generate_diagram,
        "add_elements": add_elements,
        "modify_diagram": modify_diagram
    }

    schema = [
        {
            "type": "function",
            "function": {
                "name": "generate_diagram",
                "description": "Generate or redraw a complete diagram canvas with elements. Asks for confirmation before saving to canvas.svg.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique identifier like 'rect_login', 'node_validate', 'arrow_1'"},
                                    "type": {"type": "string", "description": "Shape type, e.g. 'rectangle', 'diamond', 'circle', 'arrow'"},
                                    "label": {"type": "string", "description": "Display label or text for the element"},
                                    "color": {"type": "string", "description": "Color name or hex code"},
                                    "from": {"type": "string", "description": "Source element ID (for arrows)"},
                                    "to": {"type": "string", "description": "Target element ID (for arrows)"}
                                },
                                "required": ["id", "type"]
                            },
                            "description": "List of elements that make up the diagram."
                        }
                    },
                    "required": ["elements"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_elements",
                "description": "Add new elements (boxes, failure steps, decision branches, arrows) to the existing diagram canvas without deleting existing elements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "description": "Unique identifier"},
                                    "type": {"type": "string", "description": "Shape type ('rectangle', 'diamond', 'circle', 'arrow')"},
                                    "label": {"type": "string", "description": "Text label"},
                                    "color": {"type": "string", "description": "Color"},
                                    "from": {"type": "string", "description": "Source element ID (for arrows)"},
                                    "to": {"type": "string", "description": "Target element ID (for arrows)"}
                                },
                                "required": ["id", "type"]
                            },
                            "description": "New elements to add to the existing canvas."
                        }
                    },
                    "required": ["elements"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "modify_diagram",
                "description": "Modify properties of a specific existing element on canvas using its ID. Asks for confirmation before updating canvas.svg.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "ID of the specific element to modify."},
                        "updates": {"type": "object", "description": "Key-value dictionary of properties to update (e.g. {'color': 'amber'})."}
                    },
                    "required": ["id", "updates"]
                }
            }
        }
    ]

    return tool_map, schema