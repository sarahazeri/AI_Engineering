import json
from typing import List, Dict, Any

class Canvas:
    """State manager for holding diagram elements on the canvas."""
    def __init__(self):
        self.elements: List[Dict[str, Any]] = []

    def set_elements(self, elements: List[Dict[str, Any]]):
        """Replace all elements on canvas with a new list of elements."""
        self.elements = elements

    def update_element(self, element_id: str, updates: Dict[str, Any]) -> bool:
        """Update properties of a single element by its ID."""
        for el in self.elements:
            if el.get("id") == element_id:
                el.update(updates)
                return True
        return False

    def get_elements(self) -> List[Dict[str, Any]]:
        """Return all elements currently on the canvas."""
        return self.elements

    def clear(self):
        """Clear all elements from the canvas."""
        self.elements = []


def make_diagram_tools(canvas: Canvas):
    """
    Factory creating tools bound to a specific canvas instance.
    This pattern ensures tests in the Eval harness can use private canvas instances.
    """
    
    def generate_diagram(elements: List[Dict[str, Any]]) -> str:
        """Draw a whole diagram in one shot by replacing canvas elements."""
        try:
            canvas.set_elements(elements)
            return f"Successfully generated diagram with {len(elements)} elements."
        except Exception as e:
            return f"Error in generate_diagram: {str(e)}"

    def modify_diagram(id: str, updates: Dict[str, Any]) -> str:
        """Change one existing element by its ID without redrawing the whole canvas."""
        try:
            success = canvas.update_element(id, updates)
            if success:
                return f"Successfully updated element '{id}'."
            else:
                return f"Error: Element with id '{id}' not found on canvas."
        except Exception as e:
            return f"Error in modify_diagram: {str(e)}"

    tool_map = {
        "generate_diagram": generate_diagram,
        "modify_diagram": modify_diagram
    }

    schema = [
        {
            "type": "function",
            "function": {
                "name": "generate_diagram",
                "description": "Generate or redraw a complete diagram canvas with elements (boxes, shapes, text, arrows).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "Unique identifier like 'rect_login', 'node_validate', 'arrow_1'"
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Shape type, e.g. 'rectangle', 'diamond', 'circle', 'arrow'"
                                    },
                                    "label": {
                                        "type": "string",
                                        "description": "Display label or text for the element"
                                    },
                                    "color": {
                                        "type": "string",
                                        "description": "Color name or hex code (e.g. 'amber', 'red', '#ff0000')"
                                    },
                                    "from": {
                                        "type": "string",
                                        "description": "Source element ID (for arrows)"
                                    },
                                    "to": {
                                        "type": "string",
                                        "description": "Target element ID (for arrows)"
                                    }
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
                "name": "modify_diagram",
                "description": "Modify properties of a specific existing element on the canvas using its ID without changing other elements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "ID of the specific element to modify."
                        },
                        "updates": {
                            "type": "object",
                            "description": "Key-value dictionary of properties to update (e.g. {'color': 'amber', 'type': 'diamond'})."
                        }
                    },
                    "required": ["id", "updates"]
                }
            }
        }
    ]

    return tool_map, schema