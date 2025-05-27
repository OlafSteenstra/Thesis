import random
from latest import Vertex
from latest.Vertex import Vertex
from collections import deque
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt
import imageio.v2 as imageio

UNPROCESSED = "unprocessed"
PROCESSING = "processing"
PROCESSED = "processed"

class PhylogeneticNetwork_Viz:
    def __init__(self):
        self.vertices = {}
        self.root = None
        self.leaves = []
        self.all_states = None
        self.char_C = {}
        
        #Visualization Attributes
        self._frames = []
        self._frame_counter = 0
        self._viz_output_dir = "_algo_viz_frames"
        self._highlighted_nodes = set()
        
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
        
    def _setup_visualization(self):
        #Creates the temporary directory for frames.
        if os.path.exists(self._viz_output_dir):
            shutil.rmtree(self._viz_output_dir)
        os.makedirs(self._viz_output_dir)
        self._frames = []
        self._frame_counter = 0

    def _get_node_label(self, vertex, phase="processing"):
        label = f"{vertex.id}\n"
        if phase == "final" and vertex.final_C is not None:
            label += f"[{vertex.final_C}]"
        elif vertex.C_prime:
            state_str = str(sorted(list(vertex.C_prime)))
            max_len = 25
            if len(state_str) > max_len:
                state_str = state_str[:max_len-3] + "..."
            label += f"{state_str}"
        else:
             label += "{}"
        return label

    def _get_node_color(self, vertex):
        if vertex.id in self._highlighted_nodes:
            return 'yellow'
        elif vertex.status == PROCESSED:
            return 'lightgreen'
        elif vertex.status == PROCESSING:
             return 'orange'
        else:
            return 'lightgrey'

    def _create_visualization_frame(self, step_info="Step", phase="processing",
                                scoring_edges=None,
                                non_scoring_edges=None):
        """Creates and saves a single frame of the visualization.

        Args:
            step_info (str): Text for the plot title.
            phase (str): 'processing' or 'final' (controls label content).
            scoring_edges (list, optional): Edges contributing to score (draw red).
            non_scoring_edges (list, optional): Active edges not contributing (draw black).
            inactive_edges (list, optional): Inactive reticulation edges (draw grey/dashed).
        """
        G = nx.DiGraph()
        for vid in self.vertices:
            G.add_node(vid)

        # Determine which edges to draw if categorized lists are not provided
        # (for frames generated *before* the final scoring categorization)
        all_edges_for_layout = []
        if scoring_edges is None and non_scoring_edges is None:
            # Default behavior: categorize based on current active_parent_idx
            # for visualization *during* processing phases
            _tree_edges = []
            for u_id, u_vertex in self.vertices.items():
                for v_vertex in u_vertex.children:
                    edge = (u_id, v_vertex.id)
                    _tree_edges.append(edge)
            # In this default mode, draw active/tree edges normally, inactive ones dashed
            edges_to_draw_normal = _tree_edges
            edges_to_draw_scoring = [] # No scoring emphasis during processing
            all_edges_for_layout = edges_to_draw_normal

        else:
            # Use provided lists (for final scoring frames)
            edges_to_draw_scoring = scoring_edges if scoring_edges is not None else []
            edges_to_draw_normal = non_scoring_edges if non_scoring_edges is not None else []
            all_edges_for_layout = edges_to_draw_scoring + edges_to_draw_normal

        # --- Layout ---
        pos = None
        try:
            # Use all edges for layout calculation
            layout_G = nx.DiGraph()
            layout_G.add_nodes_from(G.nodes())
            layout_G.add_edges_from(all_edges_for_layout)
            pos = nx.nx_pydot.graphviz_layout(layout_G, prog='dot')
            
        except Exception as e:
            print(f"Graphviz layout failed ({e}). Falling back.")
            pos = nx.spring_layout(G, seed=42) # Fallback layout

        plt.figure(figsize=(12, 8))

        # --- Drawing ---
        node_labels = {vid: self._get_node_label(v, phase) for vid, v in self.vertices.items()}
        node_colors = [self._get_node_color(self.vertices[vid]) for vid in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)

        # --- Draw Edges by Category ---
        edge_alpha = 0.8
        connection_style='arc3,rad=0.1'
        arrow_style = '-|>'
        arrow_size = 15

        # Scoring Edges (Red)
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_scoring,
                            arrows=True, arrowstyle=arrow_style, arrowsize=arrow_size,
                            node_size=800, connectionstyle=connection_style,
                            edge_color='red', width=2.0, alpha=edge_alpha)

        # Non-Scoring Active/Tree Edges (Black)
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw_normal,
                            arrows=True, arrowstyle=arrow_style, arrowsize=arrow_size,
                            node_size=800, connectionstyle=connection_style,
                            edge_color='black', width=1.5, alpha=edge_alpha)

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        plt.title(f"{step_info} (Frame {self._frame_counter})")
        plt.axis('off'); plt.tight_layout()
        frame_filename = os.path.join(self._viz_output_dir, f"frame_{self._frame_counter:04d}.png")
        plt.savefig(frame_filename, dpi=100); plt.close()
        self._frames.append(frame_filename); self._frame_counter += 1
        self._highlighted_nodes.clear()

    def create_output_video(self, output_filename="algo_visualization.gif", fps=2, keep_frames=False):
        """Creates a GIF or MP4 from the saved frames."""
        if not self._frames:
            print("No frames generated to create video.")
            return

        print(f"\nCreating output: {output_filename} at {fps} FPS...")
        _, ext = os.path.splitext(output_filename)
        is_gif = ext.lower() == '.gif'

        images = [imageio.imread(frame) for frame in self._frames]

        try:
            if is_gif:
                # Duration based on fps for GIF
                duration_per_frame = 1000 / fps
                imageio.mimsave(output_filename, images, duration=duration_per_frame)
            else: # Assume video format like MP4
                # Use imageio writer
                with imageio.get_writer(output_filename, fps=fps) as writer:
                    for image in images:
                        writer.append_data(image)
            print(f"Successfully created {output_filename}")
        except Exception as e:
             print(f"Error creating output file: {e}")
             print("Ensure imageio is installed correctly. For MP4, ffmpeg backend might be needed.")

        # --- Cleanup ---
        if not keep_frames:
            print(f"Removing temporary frame directory: {self._viz_output_dir}")
            shutil.rmtree(self._viz_output_dir)
        else:
             print(f"Keeping frames in: {self._viz_output_dir}")
             
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
        self._create_visualization_frame("Initialization Complete")
        
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

        c1, c2 = vertex.children
        if not (c1.status == PROCESSED and c2.status == PROCESSED):
            return False

        self._highlighted_nodes = {vertex.id}
        step_info = f"Processing Tree/Root Node {vertex.id}"
        self._create_visualization_frame(step_info + " - Start")

        intersection = c1.C_prime.intersection(c2.C_prime)
        if intersection:
            vertex.C_prime = intersection
            if vertex.type == 'root':
                self._create_visualization_frame(step_info + " - Root Node")
                if len(vertex.C_prime) > 1:
                    vertex.C_prime = {random.choice(tuple(vertex.C_prime))}
                vertex.status = PROCESSED
                self._create_visualization_frame(step_info + " - Done")
            else:
                vertex.status = PROCESSED
                self._create_visualization_frame(step_info + " - Done")
            if len(vertex.C_prime) == 1:
                self.resolve(vertex)
        else:
            vertex.C_prime = c1.C_prime.union(c2.C_prime)
            vertex.status = PROCESSED
            self._create_visualization_frame(step_info + " - Done")
            
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

                self._highlighted_nodes = {child.id}
                step_info = f"Resolving Node {child.id}"
                self._create_visualization_frame(step_info + " - Start")

                if v_prime <= child.C_prime:
                    child.C_prime = v_prime.copy()
                else:
                    if len(child.C_prime) > 1:
                        child.C_prime = {random.choice(tuple(child.C_prime))}
                self._create_visualization_frame(step_info + " - Done")
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
                    self._highlighted_nodes = {child.id}
                    step_info = f"Final Resolving Node {child.id}"
                    self._create_visualization_frame(step_info + " - Start")
                    if {v_prime} <= child.C_prime.copy():
                        child.C_prime = {v_prime}
                        child.final_C = v_prime
                    else:
                        if len(child.C_prime) == 0:
                            print(f"Error: Child {child.id} has no C' assignment.")
                        if len(child.C_prime) > 1:
                            child.C_prime = {random.choice(tuple(child.C_prime))}
                        child.final_C = min(child.C_prime)
                    self._create_visualization_frame(step_info + " - Done")
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
                
        self._highlighted_nodes = {v1.id, v2.id, vr.id}
        step_info = f"Processing Reticulation Structure @ {vr.id}"
        self._create_visualization_frame(step_info + " - Start")
        
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
        self._highlighted_nodes = {v1.id}
        if C_prime_v2:
            vr.C_prime = C_prime_vr
            v2.C_prime = C_prime_v2
            vr.status = PROCESSED
            v2.status = PROCESSED
            self._highlighted_nodes = {v1.id, v2.id, vr.id}
        
        self._create_visualization_frame(step_info + " - Done")
        return True
    
    def _reticulation_to_tree(self, v1, v2, vr, w2, wr):
            if len(v2.parents) != 1:
                    print(f"Error: v2 ({v2.id}) has multiple parents. Cannot process.")
                    return False
            v2_parent = v2.parents[0] if v2.parents else None
            self._highlighted_nodes = {v1.id, v2.id, vr.id, w2.id, wr.id, v2_parent.id if v2_parent else None}
            self._create_visualization_frame("Pre - Reticulation to Tree")
            
            self.delete_vertex(v2.id)
            v2_parent.add_child(w2)
                     
            self._highlighted_nodes = {v1.id, vr.id, w2.id, wr.id, v2_parent.id if v2_parent else None}
            self._create_visualization_frame("In Process - Reticulation to Tree")
            
            self.delete_vertex(vr.id)
            v1.add_child(wr)
            
            self._highlighted_nodes = {v1.id, w2.id, wr.id, v2_parent.id if v2_parent else None}
            self._create_visualization_frame("Post - Reticulation to Tree")
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
        
        self._highlighted_nodes = {vr.id, v_ready.id}
        step_info = f"Processing Reticulation Workaround @ {vr.id} (using {v_ready.id})"
        self._create_visualization_frame(step_info + " - Start")
        
        vr.C_prime = w_ready.C_prime.copy() | wr.C_prime.copy()
        v_ready.C_prime = w_ready.C_prime.copy() | wr.C_prime.copy()
        
        vr.status = PROCESSED
        v_ready.status = PROCESSED
        
        self._create_visualization_frame(step_info + " - Done")
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

            # -- Priority 1: process all ready tree/root vertices -----------
            while queue_tree:
                v = queue_tree.popleft()
                in_tree_q.discard(v.id)
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
                if self._process_reticulation_structure(v1, v2, vr, w1, w2, wr):
                    progress_made = True
                    # parents of v1/v2 might now be ready tree nodes
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
                    #TODO: Might have to be the other way around?
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

        self._create_visualization_frame("After Final Cleanup", phase="final",
                                         scoring_edges=scoring_edges_final,
                                         non_scoring_edges=non_scoring_edges_final)

        total_score = len(scoring_edges_final)

        self._create_visualization_frame(f"Final Score: {total_score}", phase="final",
                                         scoring_edges=scoring_edges_final,
                                         non_scoring_edges=non_scoring_edges_final)

        final_assignment = {vid: v.final_C for vid, v in self.vertices.items()}
        return total_score, final_assignment
    
    def run_approximation(self, char_C, output_video="algo_visualization.gif", fps=1, keep_frames=True):
        score, assignment = None, None
        try:
            self._setup_visualization()
            self._initialize_algorithm(char_C)
            self._iterative_processing()
            pre_score, assignment = self._finalize_assignments_and_score()
            print(f"Pre-Score: {pre_score}")
            #WHAT IF I RUN ALGO AGAIN?
            self._initialize_algorithm(char_C)
            self._iterative_processing()
            score, assignment = self._finalize_assignments_and_score()
            self.create_output_video(output_video, fps, keep_frames)
            return pre_score, score, assignment
        except Exception as e:
            print(f"\n--- Algorithm Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.create_output_video(output_video, fps, keep_frames)
            return None, None