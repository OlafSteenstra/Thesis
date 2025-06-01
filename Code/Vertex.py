UNPROCESSED = "unprocessed"
PROCESSING = "processing"
PROCESSED = "processed"

class Vertex:
    def __init__(self, id, type, input_C=None):
        """Initializes a Vertex object with its ID, type, and processing states."""
        self.id = id
        if type not in ['leaf', 'tree', 'reticulation', 'root']:
            raise ValueError(f"Invalid vertex type: {type}")
        self.type = type
        self.children = []
        self.parents = []
        self.status = UNPROCESSED
        self.input_C = input_C
        self.C_prime = set()
        self.final_C = None

    def add_child(self, child_vertex):
        """Adds a child vertex and establishes bidirectional parent-child links."""
        if child_vertex not in self.children:
            self.children.append(child_vertex)
        if self not in child_vertex.parents:
            child_vertex.parents.append(self)

    def get_other_child(self, known_child):
        """Returns the other child of a tree node, if it exists."""
        if len(self.children) != 2: return None
        if self.children[0] == known_child:
            return self.children[1]
        elif self.children[1] == known_child:
            return self.children[0]
        return None