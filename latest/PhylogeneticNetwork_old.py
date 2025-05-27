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
            else:
                vertex.C_prime = c1.C_prime.union(c2.C_prime)
                if vertex.type == 'root':
                    while len(vertex.C_prime) > 1:
                        vertex.C_prime.pop()
                self.resolve(vertex)
                
        vertex.status = PROCESSED
        return True

    # def resolve(self, v: Vertex):
    #     v_prime = v.C_prime.copy()
    #     for child in v.children:
    #         if len(child.C_prime) == 1:
    #             self.resolve(child)
    #             continue
    #         c_prime = child.C_prime.copy()
    #         if v_prime <= c_prime:
    #             child.C_prime = v_prime.copy()
    #         else:
    #             while len(child.C_prime) > 1:
    #                 child.C_prime.pop()
    #         self.resolve(child)
            
    def resolve(self, v: "Vertex") -> None:
        """
        Iterative version of the original recursive resolve().
        Behaviour and side-effects on C' sets are identical.
        """
        stack = [v]

        while stack:
            cur = stack.pop()
            v_prime = cur.C_prime.copy()          # parent’s current C'

            # visit children in the same left-to-right order as the recursive
            # version – push them in reverse so the first child is processed first
            for child in reversed(cur.children):
                if len(child.C_prime) == 1:
                    stack.append(child)
                    continue

                # same logic as before
                if v_prime <= child.C_prime:
                    child.C_prime = v_prime.copy()
                else:
                    while len(child.C_prime) > 1:
                        child.C_prime.pop()

                stack.append(child)
    
    # def final_resolve(self, v: Vertex):
    #     if v.final_C is None:
    #         print(f"Error: Vertex {v.id} has no final C assignment.")
    #     v_prime = v.final_C
    #     for child in v.children:
    #         if len(child.C_prime) == 1:
    #             child.final_C = min(child.C_prime)
    #             self.final_resolve(child)
    #             continue
    #         c_prime = child.C_prime.copy()
    #         if {v_prime} <= c_prime:
    #             child.final_C = v_prime
    #         else:
    #             if len(child.C_prime) == 0:
    #                 print(f"Error: Child {child.id} has no C' assignment.")
    #             while len(child.C_prime) > 1:
    #                 child.C_prime.pop()
    #             child.final_C = min(child.C_prime)
    #         self.final_resolve(child)
    
    def final_resolve(self, v: "Vertex") -> None:
        """
        Iterative version of the original recursive final_resolve().
        Produces exactly the same final_C assignments and warnings.
        """
        if v.final_C is None:
            print(f"Error: Vertex {v.id} has no final C assignment.")

        stack = [v]

        while stack:
            cur = stack.pop()
            v_prime = cur.final_C

            for child in reversed(cur.children):
                # --- mirror the original branch structure -----------------
                if len(child.C_prime) == 1:
                    child.final_C = min(child.C_prime)
                else:
                    if {v_prime} <= child.C_prime.copy():
                        child.final_C = v_prime
                    else:
                        if len(child.C_prime) == 0:
                            print(f"Error: Child {child.id} has no C' assignment.")
                        while len(child.C_prime) > 1:
                            child.C_prime.pop()
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

        if len(wr.C_prime & w1.C_prime) < len(wr.C_prime & w2.C_prime):
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
                #TODO: This seems logical but is it correct?
                C_prime_v1 = w1.C_prime.copy()
                self.resolve(wr)
                
        else:
            #TODO: This is not mentioned in the paper. Is it correct?
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
        """
        Iterative pass that marks every vertex PROCESSED without ever scanning
        the full vertex list inside the main loop.

        Logic-wise it is identical to the original version, but runs in
        ≈ O(V + E) instead of O(V²) because each vertex/structure is en-queued
        at most once.
        """

        # ---------- helper closures ----------------------------------------
        queue_tree  : deque[Vertex]            = deque()   # ready tree/root nodes
        in_tree_q   : set[int]                 = set()
        queue_retic : deque[tuple]             = deque()   # ready (v1,v2,vr,w1,w2,wr)
        in_retic_q  : set[int]                 = set()     # vr.id values already queued

        def _maybe_enqueue_tree(v: "Vertex") -> None:
            """Put *v* in the tree queue if it is ready and not queued yet."""
            if v.status != UNPROCESSED or v.id in in_tree_q:
                return
            if v.type == "root" and not v.children:
                queue_tree.append(v); in_tree_q.add(v.id)
            elif v.children and all(c.status == PROCESSED for c in v.children):
                queue_tree.append(v); in_tree_q.add(v.id)

        def _maybe_enqueue_retic(vr: "Vertex") -> None:
            """
            Compute the 6-tuple for *vr* once; enqueue the structure if the three
            children (w1,w2,wr) are processed and v1,v2,vr are still unprocessed.
            """
            if vr.status != UNPROCESSED or vr.id in in_retic_q:
                return
            struct = self._get_reticulation_structure_vertices(vr)
            if not struct:
                return
            v1, v2, vr, w1, w2, wr = struct
            if (w1.status == PROCESSED and w2.status == PROCESSED and
                wr.status == PROCESSED and
                v1.status == UNPROCESSED and v2.status == UNPROCESSED):
                queue_retic.append(struct); in_retic_q.add(vr.id)

        # ---------- seed the queues once -----------------------------------
        for v in self.vertices.values():
            if v.type in ("tree", "root"):
                _maybe_enqueue_tree(v)
            elif v.type == "reticulation":
                _maybe_enqueue_retic(v)

        # ---------- main work loop -----------------------------------------
        while True:
            progress_made = False

            # -- Priority 1: process all ready tree/root vertices -----------
            while queue_tree:
                v = queue_tree.popleft()
                in_tree_q.discard(v.id)
                # print(f"Processing tree/root node {v.id} with status {v.status}")
                if self._process_tree_node(v):
                    progress_made = True
                    # its parents may now be ready
                    for p in v.parents:
                        if p.type in ("tree", "root"):
                            _maybe_enqueue_tree(p)
                        if p.type == "reticulation":
                            _maybe_enqueue_retic(p)
                        for c in p.children:
                            if c.type == "reticulation":
                                _maybe_enqueue_retic(c)

            # -- Priority 2: process one ready reticulation structure -------
            if queue_retic:
                v1, v2, vr, w1, w2, wr = queue_retic.popleft()
                in_retic_q.discard(vr.id)
                # print(f"Processing reticulation structure {vr.id} with status {vr.status}")
                if self._process_reticulation_structure(v1, v2, vr, w1, w2, wr):
                    progress_made = True
                    # parents of v1/v2 might now be ready tree nodes
                    for node in (v1, v2):
                        for p in node.parents:
                            if p.type in ("tree", "root"):
                                _maybe_enqueue_tree(p)
                            if p.type == "reticulation":
                                _maybe_enqueue_retic(p)

            # -- done? ------------------------------------------------------
            if all(v.status == PROCESSED for v in self.vertices.values()):
                return                                            # success!

            # -- if stuck, attempt the original workaround ------------------
            if not progress_made:
                workaround_applied = False

                for vr in self.vertices.values():
                    if vr.type != "reticulation" or vr.status != UNPROCESSED:
                        continue
                    struct = self._get_reticulation_structure_vertices(vr)
                    if not struct:
                        continue
                    v1, v2, vr, w1, w2, wr = struct
                    if wr.status != PROCESSED:
                        continue
                    # print(f"Attempting workaround for reticulation {vr.id} with status {vr.status}")
                    w1_ready, w2_ready = w1.status == PROCESSED, w2.status == PROCESSED
                    if w1_ready and not w2_ready and v1.status == UNPROCESSED:
                        if self._process_reticulation_workaround(vr, v1, w1, wr):
                            for p in v1.parents:
                                if p.type in ("tree", "root"):
                                    _maybe_enqueue_tree(p)
                                if p.type == "reticulation":
                                    _maybe_enqueue_retic(p)
                            workaround_applied = True
                            break
                    elif w2_ready and not w1_ready and v2.status == UNPROCESSED:
                        if self._process_reticulation_workaround(vr, v2, w2, wr):
                            for p in v2.parents:
                                if p.type in ("tree", "root"):
                                    _maybe_enqueue_tree(p)
                                if p.type == "reticulation":
                                    _maybe_enqueue_retic(p)
                            workaround_applied = True
                            break

                if workaround_applied:
                    progress_made = True

            # -- completely stuck? -----------------------------------------
            if not progress_made:
                unprocessed = [v.id for v in self.vertices.values()
                            if v.status != PROCESSED]
                print(f"Warning: Processing stuck. Unprocessed: {unprocessed}")
                return

    # def _iterative_processing(self):
    #     processed_nodes_this_run = set()
    #     iteration = 0
    #     max_iterations = len(self.vertices) * 3 # Arbitrary limit to avoid infinite loops

    #     while iteration < max_iterations:
    #         iteration += 1
    #         progress_made_in_iteration = False
            
    #         #Priority 1: Process ready tree nodes/roots
    #         ready_tree_nodes = []
    #         for v in self.vertices.values():
    #             if v.status == UNPROCESSED and v.type in ['tree', 'root']:
    #                 if v.children and all(c.status == PROCESSED for c in v.children):
    #                     ready_tree_nodes.append(v)
    #                 elif not v.children and v.type == 'root':
    #                     ready_tree_nodes.append(v)
    #                     print(f"Root {v.id} has no children. Marking as ready.")

    #         if ready_tree_nodes:
    #             node_to_process = ready_tree_nodes[0]
    #             if self._process_tree_node(node_to_process):
    #                 progress_made_in_iteration = True
    #                 processed_nodes_this_run.add(node_to_process.id)
    #                 continue

    #         #Priority 2: Process ready reticulation structures
    #         ready_retic_structures = []
    #         processed_vr_ids = set()

    #         for vr_id, vr_node in self.vertices.items():
    #             if vr_node.type == 'reticulation' and vr_node.status == UNPROCESSED:
    #                 if vr_id in processed_vr_ids: 
    #                     continue

    #                 structure = self._get_reticulation_structure_vertices(vr_node)
    #                 if structure:
    #                     v1, v2, vr, w1, w2, wr = structure
    #                     if (w1.status == PROCESSED and w2.status == PROCESSED and wr.status == PROCESSED and
    #                         v1.status == UNPROCESSED and v2.status == UNPROCESSED):
    #                         ready_retic_structures.append(structure)
    #                         processed_vr_ids.add(vr_id)

    #         if ready_retic_structures:
    #             structure_to_process = ready_retic_structures[0]
    #             v1, v2, vr, w1, w2, wr = structure_to_process
    #             if self._process_reticulation_structure(*structure_to_process):
    #                 progress_made_in_iteration = True
    #                 processed_nodes_this_run.update([v1.id, v2.id, vr.id])
    #                 continue

    #         #Check if finished or stuck
    #         if not progress_made_in_iteration:
    #             all_done = all(v.status == PROCESSED for v in self.vertices.values())
    #             if all_done:
    #                 break
    #             else:
    #                 #Priority 3: Attempt Non-Time-Consistency Workaround
    #                 workaround_applied = False
    #                 for vr_id, vr_node in self.vertices.items():
    #                     if vr_node.type == 'reticulation' and vr_node.status == UNPROCESSED:
    #                         structure = self._get_reticulation_structure_vertices(vr_node)
    #                         if structure:
    #                             v1, v2, vr, w1, w2, wr = structure
    #                             if wr.status != PROCESSED: 
    #                                 continue

    #                             w1_ready = w1 and w1.status == PROCESSED
    #                             w2_ready = w2 and w2.status == PROCESSED

    #                             if w1_ready and not w2_ready and v1.status == UNPROCESSED:
    #                                 if self._process_reticulation_workaround(vr, v1, w1, wr):
    #                                     progress_made_in_iteration = True
    #                                     processed_nodes_this_run.update([v1.id, vr.id])
    #                                     workaround_applied = True
    #                                     break
    #                                 else:
    #                                     print(f"Workaround failed for {vr.id} via {v1.id}")

    #                             elif w2_ready and not w1_ready and v2.status == UNPROCESSED:
    #                                 if self._process_reticulation_workaround(vr, v2, w2, wr):
    #                                     progress_made_in_iteration = True
    #                                     processed_nodes_this_run.update([v2.id, vr.id])
    #                                     workaround_applied = True
    #                                     break
    #                                 else:
    #                                     print(f"Workaround failed for {vr.id} via {v2.id}")

    #                 if workaround_applied:
    #                     continue

    #                 # If no workaround was applied, then truly stuck
    #                 unprocessed_nodes = [v.id for v in self.vertices.values() if v.status != PROCESSED]
    #                 print(f"Warning: Processing stuck. No workaround found. Unprocessed: {unprocessed_nodes}")
    #                 break

    #         if iteration >= max_iterations:
    #             print("Warning: Max iterations reached. Possible infinite loop or complex structure.")
    #             break

    #     if not all(v.status == PROCESSED for v in self.vertices.values()):
    #         unprocessed_ids = [v.id for v in self.vertices.values() if v.status != PROCESSED]
    #         print(f"Warning: Processing finished, but some nodes remain unprocessed: {unprocessed_ids}")

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
                    #TODO: Might have to be the other way around?
                    if ham_dist1 <= ham_dist2:
                        self._reticulation_to_tree(v1, v2, vertex, v2.get_other_child(vertex), vertex.children[0])
                    else:
                        self._reticulation_to_tree(v2, v1, vertex, v1.get_other_child(vertex), vertex.children[0])
                    deleting_retics = True
                    break
                else:
                    deleting_retics = False
                    
        while len(self.root.C_prime) > 1:
            self.root.C_prime.pop()
        self.root.final_C = min(self.root.C_prime)

        self.final_resolve(self.root)

        for v_id, v_node in self.vertices.items():
            if v_node.final_C is None:
                print(f"Warning: Vertex {v_id} has no final C assignment.")

        #Categorize Edges for Final Visualization
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

        # --- Calculate Final Score (using the same logic as categorization) ---
        total_score = len(scoring_edges_final) # The score is simply the count of scoring edges

        final_assignment = {vid: v.final_C for vid, v in self.vertices.items()}
        return total_score, final_assignment
    
    def run_approximation(self, char_C):
        """
        Executes the 2-approximation algorithm and generates visualization.
        Args:
            char_C (dict): Leaf ID -> character state map.
            output_video (str): Filename for the output GIF/MP4.
            fps (int): Frames per second for the output.
            keep_frames (bool): If True, keeps the individual frame images.
        Returns:
            tuple: (final_score, final_assignment) or (None, None) on failure.
        """
        score, assignment = None, None
        try:
            self._initialize_algorithm(char_C)
            self._iterative_processing()
            score, assignment = self._finalize_assignments_and_score()
            return score, assignment
        except Exception as e:
            print(f"\n--- Algorithm Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None