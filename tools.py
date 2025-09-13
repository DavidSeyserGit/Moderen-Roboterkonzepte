from rag import RAGManager

rag = RAGManager()

def get_status(dummy: str = "") -> str:
    """Check system status."""
    return "System is up and running ✅"

def retrieve_places(query: str) -> str:
    """Retrieve matching places from the knowledge base."""
    name = rag.retrieve_location(query)
    coordinates = rag.get_location_coords(name)
    return name, coordinates


def execute_goal(location_name: str) -> str:
    """Simulate sending robot to a named location."""
    loc = rag.get_location_coords(location_name)
    if not loc:
        return f"Unknown location: {location_name}"
    return f"Simulated move to {loc['name']} at coords=({loc['x']}, {loc['y']}, {loc['theta']})"