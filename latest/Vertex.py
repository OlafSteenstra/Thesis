UNPROCESSED = "unprocessed"
PROCESSING = "processing"
PROCESSED = "processed"

class Vertex:
    def __init__(self, id, type, input_C=None):
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
        if child_vertex not in self.children:
            self.children.append(child_vertex)
        if self not in child_vertex.parents:
            child_vertex.parents.append(self)

    def __repr__(self):
        parent_ids = [p.id for p in self.parents]
        child_ids = [c.id for c in self.children]
        return (f"Vertex(id={self.id}, type={self.type}, status={self.status}, "
                f"C'={self.C_prime}, final_C={self.final_C}, "
                f"parents={parent_ids}, children={child_ids})")

    def get_other_child(self, known_child):
        if len(self.children) != 2: return None
        if self.children[0] == known_child:
            return self.children[1]
        elif self.children[1] == known_child:
            return self.children[0]
        return None