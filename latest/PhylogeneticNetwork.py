import random
from latest import Vertex
from latest.Vertex import Vertex
from collections import deque

UNPROCESSED = "unprocessed"
PROCESSING = "processing"
PROCESSED = "processed"

class PhylogeneticNetwork:
    def __init__(self):
        self.vertices = {}
        self.root = None
        self.leaves = []
        self.all_states = None
        self.char_C = {}
        
    def add_vertex(self, vertex):
        if vertex.id in self.vertices:
            raise ValueError(f"Vertex with ID {vertex.id} already exists.")
        self.vertices[vertex.id] = vertex
        if vertex.type == 'root':
            if self.root is not None:
                raise ValueError("Network cannot have multiple roots.")
            self.root = vertex
        elif vertex.type == 'leaf':
            self.leaves.append(vertex)
    
    def delete_vertex(self, vertex_id):
        if vertex_id not in self.vertices:
            raise ValueError(f"Vertex with ID {vertex_id} does not exist.")
        vertex = self.vertices[vertex_id]
        for parent in vertex.parents:
            parent.children.remove(vertex)
        for child in vertex.children:
            child.parents.remove(vertex)
        del self.vertices[vertex_id]
        if vertex.type == 'root':
            self.root = None
            print("Warning: Deleted root vertex. No root in network.")
        elif vertex.type == 'leaf':
            self.leaves.remove(vertex)
            print(f"Warning: Deleted leaf vertex {vertex_id}.")

    def add_edge(self, parent_id, child_id):
        if parent_id not in self.vertices or child_id not in self.vertices:
            raise ValueError(f"Parent ({parent_id}) or Child ({child_id}) vertex not found.")
        parent_vertex = self.vertices[parent_id]
        child_vertex = self.vertices[child_id]
        parent_vertex.add_child(child_vertex)
             
    def _initialize_algorithm(self, char_C):
        self.char_C = char_C
        self.all_states = set(char_C.values())

        for vertex in self.vertices.values():
            vertex.status = UNPROCESSED
            vertex.C_prime = set()
            vertex.final_C = None

            if vertex.type == 'leaf':
                leaf_id = vertex.id
                if leaf_id not in char_C:
                    raise ValueError(f"Input character C missing for leaf {leaf_id}")
                state = char_C[leaf_id]
                vertex.input_C = state
                vertex.C_prime = {state}
                vertex.status = PROCESSED
        
    def _process_tree_node(self, vertex):
        if vertex.status != UNPROCESSED: 
            return False
        if vertex.type not in ['tree', 'root']: 
            return False

        if not vertex.children:
            if vertex.type == 'root':
                vertex.status = PROCESSED
                return True
            else: 
                return False
      
        if len(vertex.children) != 2:
            print(f"Warning: Tree/Root node {vertex.id} has {len(vertex.children)} children.")
            return False

        else:
            c1, c2 = vertex.children
            if not (c1.status == PROCESSED and c2.status == PROCESSED):
                return False

            intersection = c1.C_prime.intersection(c2.C_prime)
            if intersection:
                vertex.C_prime = intersection
                if vertex.type == 'root':
                    if len(vertex.C_prime) > 1:
                        vertex.C_prime = {random.choice(tuple(vertex.C_prime))}
                if len(vertex.C_prime) == 1:
                    self.resolve(vertex)
            else:
                vertex.C_prime = c1.C_prime.union(c2.C_prime)
                
        vertex.status = PROCESSED
        return True
            
    def resolve(self, v: "Vertex") -> None:
        stack = [v]

        while stack:
            cur = stack.pop()
            v_prime = cur.C_prime.copy()

            for child in reversed(cur.children):
                if len(child.C_prime) == 1:
                    stack.append(child)
                    continue

                if v_prime <= child.C_prime:
                    child.C_prime = v_prime.copy()
                else:
                    if len(child.C_prime) > 1:
                        child.C_prime = {random.choice(tuple(child.C_prime))}

                stack.append(child)
    
    def final_resolve(self, v: "Vertex") -> None:
        if v.final_C is None:
            print(f"Error: Vertex {v.id} has no final C assignment.")

        stack = [v]

        while stack:
            cur = stack.pop()
            v_prime = cur.final_C

            for child in reversed(cur.children):
                if len(child.C_prime) == 1:
                    child.final_C = min(child.C_prime)
                else:
                    if {v_prime} <= child.C_prime.copy():
                        child.final_C = v_prime
                    else:
                        if len(child.C_prime) == 0:
                            print(f"Error: Child {child.id} has no C' assignment.")
                        if len(child.C_prime) > 1:
                            child.C_prime = {random.choice(tuple(child.C_prime))}
                        child.final_C = min(child.C_prime)

                stack.append(child)

        
    def _get_reticulation_structure_vertices(self, vr):
        #Helper to find v1, v2, vr, w1, w2, wr.
        if vr.type != 'reticulation' or len(vr.parents) != 2 or len(vr.children) != 1:
            return None

        v1, v2 = vr.parents
        wr = vr.children[0]
        w1 = v1.get_other_child(vr)
        w2 = v2.get_other_child(vr)

        if w1 is None:
            print(f"Error: Parent v1 ({v1.id}) type 'tree' has only child vr ({vr.id}). Not binary?")
            return None

        if w2 is None:
            print(f"Error: Parent v2 ({v2.id}) type 'tree' has only child vr ({vr.id}). Not binary?")
            return None

        if len(wr.C_prime & w1.C_prime) < len(wr.C_prime & w2.C_prime) or (len(wr.C_prime & w1.C_prime) == len(wr.C_prime & w2.C_prime) and random.random() < 0.5):
            return v2, v1, vr, w2, w1, wr
        return v1, v2, vr, w1, w2, wr
 
    def _process_reticulation_structure(self, v1, v2, vr, w1, w2, wr):
        if not (w1 and w2 and wr):
             print(f"Error: Missing vertices for structure around {vr.id}. Cannot process.")
             return False
        if not (w1.status == PROCESSED and w2.status == PROCESSED and wr.status == PROCESSED):
            print(f"Error: Called _process_reticulation_structure for {vr.id} but prerequisites w1/w2/wr not processed.")
            return False
        if not (v1.status == UNPROCESSED and v2.status == UNPROCESSED and vr.status == UNPROCESSED):
            print(f"Warning: Processing reticulation structure for {vr.id}, but v1/v2/vr not all UNPROCESSED (Statuses: {v1.status}/{v2.status}/{vr.status}). Skipping.")
            return False
                
        C_prime_v1 = set()
        C_prime_v2 = None
        if len(w1.C_prime & wr.C_prime) > 0:
            
            if not self._reticulation_to_tree(v1, v2, vr, w2, wr):
                print(f"Error: Failed to convert reticulation to tree for {vr.id}.")
                return False
            
            C_prime_v1 = wr.C_prime.copy()
            if wr.C_prime < w1.C_prime:
                w1.C_prime = wr.C_prime.copy()
                self.resolve(w1)
            elif w1.C_prime < wr.C_prime:
                wr.C_prime = w1.C_prime.copy()
                C_prime_v1 = w1.C_prime.copy()
                self.resolve(wr)
                
        else:
            C_prime_vr = w1.C_prime.copy() | wr.C_prime.copy()
            C_prime_v1 = w1.C_prime.copy() | wr.C_prime.copy()
            C_prime_v2 = w2.C_prime.copy() | wr.C_prime.copy()
        
        v1.C_prime = C_prime_v1
        v1.status = PROCESSED
        if C_prime_v2:
            vr.C_prime = C_prime_vr
            v2.C_prime = C_prime_v2
            vr.status = PROCESSED
            v2.status = PROCESSED
        
        return True
    
    def _reticulation_to_tree(self, v1, v2, vr, w2, wr):
            if len(v2.parents) != 1:
                    print(f"Error: v2 ({v2.id}) has multiple parents. Cannot process.")
                    return False
            v2_parent = v2.parents[0] if v2.parents else None
            
            self.delete_vertex(v2.id)
            v2_parent.add_child(w2)
            self.delete_vertex(vr.id)
            v1.add_child(wr)
            
            return True
                
    def _process_reticulation_workaround(self, vr, v_ready, w_ready, wr):
        if vr.status != UNPROCESSED or v_ready.status != UNPROCESSED:
             print(f"Error: Workaround called on non-unprocessed vr ({vr.status}) or v_ready ({v_ready.status}).")
             return False
        if w_ready.status != PROCESSED:
             print(f"Error: Workaround called but w_ready ({w_ready.id}) is not processed.")
             return False
        if not wr or wr.status != PROCESSED:
             print(f"Error: Workaround called but wr child ({vr.children[0].id if vr.children else 'None'}) is not processed.")
             return False
                        
        vr.C_prime = w_ready.C_prime.copy() | wr.C_prime.copy()
        v_ready.C_prime = w_ready.C_prime.copy() | wr.C_prime.copy()
        
        vr.status = PROCESSED
        v_ready.status = PROCESSED
        return True
    
    def _iterative_processing(self) -> None:
        queue_tree  : deque[Vertex]            = deque()
        in_tree_q   : set[int]                 = set()
        queue_retic : deque[tuple]             = deque()
        in_retic_q  : set[int]                 = set()
        retic_workaround : set[Vertex] = set()
        
        def _maybe_enqueue_tree(v: "Vertex") -> None:
            if v.status != UNPROCESSED or v.id in in_tree_q:
                return
            if v.type == "root" and not v.children:
                queue_tree.append(v); in_tree_q.add(v.id)
            elif v.children and all(c.status == PROCESSED for c in v.children):
                queue_tree.append(v); in_tree_q.add(v.id)

        def _maybe_enqueue_retic(vr: "Vertex") -> None:
            if vr.status != UNPROCESSED or vr.id in in_retic_q:
                return
            struct = self._get_reticulation_structure_vertices(vr)
            if not struct:
                return
            v1, v2, vr, w1, w2, wr = struct
            if (wr.status == PROCESSED and
                v1.status == UNPROCESSED and v2.status == UNPROCESSED):
                if (w1.status == PROCESSED and w2.status == PROCESSED):
                    queue_retic.append(struct); in_retic_q.add(vr.id)
                elif (w1.status == PROCESSED or w2.status == PROCESSED):
                    retic_workaround.add(vr)

        for v in self.vertices.values():
            if v.type in ("tree", "root"):
                _maybe_enqueue_tree(v)
            elif v.type == "reticulation":
                _maybe_enqueue_retic(v)

        while True:
            progress_made = False
            
            while queue_tree:
                v = queue_tree.popleft()
                in_tree_q.discard(v.id)
                if self._process_tree_node(v):
                    progress_made = True
                    for p in v.parents:
                        if p.type in ("tree", "root"):
                            _maybe_enqueue_tree(p)
                        if p.type == "reticulation":
                            _maybe_enqueue_retic(p)
                        for c in p.children:
                            if c.type == "reticulation":
                                _maybe_enqueue_retic(c)

            if queue_retic:
                v1, v2, vr, w1, w2, wr = queue_retic.popleft()
                in_retic_q.discard(vr.id)
                if self._process_reticulation_structure(v1, v2, vr, w1, w2, wr):
                    progress_made = True
                    for node in (v1, v2):
                        for p in node.parents:
                            if p.type in ("tree", "root"):
                                _maybe_enqueue_tree(p)
                            if p.type == "reticulation":
                                _maybe_enqueue_retic(p)
                            for c in p.children:
                                if c.type == "reticulation":
                                    _maybe_enqueue_retic(c)

            if all(v.status == PROCESSED for v in self.vertices.values()):
                return

            if not progress_made:
                workaround_applied = False
                to_remove : set[Vertex] = set()
                to_check : set[Vertex] = set()
                for vr in retic_workaround:
                    if vr.type != "reticulation" or vr.status != UNPROCESSED:
                        to_remove.add(vr)
                        continue
                    struct = self._get_reticulation_structure_vertices(vr)
                    if not struct:
                        to_remove.add(vr)
                        continue
                    v1, v2, vr, w1, w2, wr = struct
                    if wr.status != PROCESSED:
                        to_remove.add(vr)
                        continue
                    w1_ready, w2_ready = w1.status == PROCESSED, w2.status == PROCESSED
                    if w1_ready and not w2_ready and v1.status == UNPROCESSED:
                        if self._process_reticulation_workaround(vr, v1, w1, wr):
                            for p in v1.parents:
                                to_check.add(p)
                            workaround_applied = True
                            to_remove.add(vr)
                    elif w2_ready and not w1_ready and v2.status == UNPROCESSED:
                        if self._process_reticulation_workaround(vr, v2, w2, wr):
                            for p in v2.parents:
                                to_check.add(p)
                            workaround_applied = True
                            to_remove.add(vr)

                if workaround_applied:
                    retic_workaround.difference_update(to_remove)
                    for p in to_check:
                        if p.type in ("tree", "root"):
                            _maybe_enqueue_tree(p)
                        if p.type == "reticulation":
                            _maybe_enqueue_retic(p)
                        for c in p.children:
                            if c.type == "reticulation":
                                _maybe_enqueue_retic(c)
                    progress_made = True

            if not progress_made:
                unprocessed = [v.id for v in self.vertices.values()
                            if v.status != PROCESSED]
                print(f"Warning: Processing stuck. Unprocessed: {unprocessed}")
                return
    
    def _finalize_assignments_and_score(self):
        if self.root.status != PROCESSED:
            print(f"Error: Root {self.root.id} was not processed. Status: {self.root.status}. Cannot finalize.")
            return None, None

        if not self.root.C_prime:
            print(f"Warning: Root {self.root.id} C' is empty after processing")
            return None, None

        deleting_retics = True
        while deleting_retics:
            for vertex in self.vertices.values():
                if vertex.type == 'reticulation':
                    v1, v2 = vertex.parents
                    ham_dist1 = None
                    ham_dist2 = None
                    if v1.C_prime == vertex.C_prime:
                        ham_dist1 = 0
                    else:
                        ham_dist1 = 1
                    if v2.C_prime == vertex.C_prime:
                        ham_dist2 = 0
                    else:
                        ham_dist2 = 1
                    if ham_dist1 < ham_dist2 or (ham_dist1 == ham_dist2 and random.random() < 0.5):
                        self._reticulation_to_tree(v1, v2, vertex, v2.get_other_child(vertex), vertex.children[0])
                    else:
                        self._reticulation_to_tree(v2, v1, vertex, v1.get_other_child(vertex), vertex.children[0])
                    deleting_retics = True
                    break
                else:
                    deleting_retics = False
                    
        if len(self.root.C_prime) > 1:
            self.root.C_prime = {random.choice(tuple(self.root.C_prime))}
        self.root.final_C = min(self.root.C_prime)

        self.final_resolve(self.root)

        for v_id, v_node in self.vertices.items():
            if v_node.final_C is None:
                print(f"Warning: Vertex {v_id} has no final C assignment.")          

        scoring_edges_final = []
        non_scoring_edges_final = []
        processed_edges_viz = set()

        for u_id, u_vertex in self.vertices.items():
            if u_vertex.final_C is None: 
                continue
            for v_vertex in u_vertex.children:
                edge = (u_id, v_vertex.id)
                if edge in processed_edges_viz: 
                    continue
                if v_vertex.final_C is None: 
                    continue

                is_scoring = False

                if u_vertex.final_C != v_vertex.final_C:
                    is_scoring = True

                if is_scoring:
                    scoring_edges_final.append(edge)
                else:
                    non_scoring_edges_final.append(edge)

                processed_edges_viz.add(edge)

        total_score = len(scoring_edges_final)

        final_assignment = {vid: v.final_C for vid, v in self.vertices.items()}
        return total_score, final_assignment
    
    def run_approximation(self, char_C):
        score, assignment = None, None
        try:
            self._initialize_algorithm(char_C)
            self._iterative_processing()
            pre_score, assignment = self._finalize_assignments_and_score()
            self._initialize_algorithm(char_C)
            self._iterative_processing()
            score, assignment = self._finalize_assignments_and_score()
            return pre_score, score, assignment
        except Exception as e:
            print(f"Algorithm Failed")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None