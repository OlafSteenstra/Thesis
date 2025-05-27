import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from latest import Vertex

UNPROCESSED = "unprocessed"
PROCESSING = "processing"
PROCESSED = "processed"

class PhylogeneticNetwork_Viz2:
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
        self._setup_visualization()
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

        self._highlighted_nodes = {vertex.id}
        step_info = f"Processing Tree/Root Node {vertex.id}"
        self._create_visualization_frame(step_info + " - Start")
        
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
        self._create_visualization_frame(step_info + " - Done")
        return True

    def resolve(self, v: Vertex):
        v_prime = v.C_prime.copy()
        for child in v.children:
            if len(child.C_prime) == 1:
                self.resolve(child)
                continue
            step_info = f"Resolving Node {child.id}"
            self._highlighted_nodes = {child.id}
            self._create_visualization_frame(step_info + " - Start")
            c_prime = child.C_prime.copy()
            if v_prime <= c_prime:
                child.C_prime = v_prime.copy()
            else:
                while len(child.C_prime) > 1:
                    child.C_prime.pop()
            self._create_visualization_frame(step_info + " - Done")
            self.resolve(child)
    
    def final_resolve(self, v: Vertex):
        if v.final_C is None:
            print(f"Error: Vertex {v.id} has no final C assignment.")
        v_prime = v.final_C
        for child in v.children:
            if len(child.C_prime) == 1:
                child.final_C = min(child.C_prime)
                self.final_resolve(child)
                continue
            step_info = f"Final Resolving Node {child.id}"
            self._highlighted_nodes = {child.id}
            self._create_visualization_frame(step_info + " - Start")
            c_prime = child.C_prime.copy()
            if {v_prime} <= c_prime:
                child.final_C = v_prime
            else:
                if len(child.C_prime) == 0:
                    print(f"Error: Child {child.id} has no C' assignment.")
                while len(child.C_prime) > 1:
                    child.C_prime.pop()
                child.final_C = min(child.C_prime)
            self._create_visualization_frame(step_info + " - Done")
            self.final_resolve(child)
        
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
        return True
    
    def _iterative_processing(self):
        processed_nodes_this_run = set()
        iteration = 0
        max_iterations = len(self.vertices) * 3 # Arbitrary limit to avoid infinite loops

        while iteration < max_iterations:
            iteration += 1
            progress_made_in_iteration = False
            print(f"\nIteration {iteration}:")
            
            #Priority 1: Process ready tree nodes/roots
            ready_tree_nodes = []
            for v in self.vertices.values():
                if v.status == UNPROCESSED and v.type in ['tree', 'root']:
                    if v.children and all(c.status == PROCESSED for c in v.children):
                        ready_tree_nodes.append(v)
                    elif not v.children and v.type == 'root':
                        ready_tree_nodes.append(v)
                        print(f"Root {v.id} has no children. Marking as ready.")

            if ready_tree_nodes:
                node_to_process = ready_tree_nodes[0]
                print(f"Processing Tree/Root Node {node_to_process.id}")
                if self._process_tree_node(node_to_process):
                    progress_made_in_iteration = True
                    processed_nodes_this_run.add(node_to_process.id)
                    continue

            #Priority 2: Process ready reticulation structures
            ready_retic_structures = []
            processed_vr_ids = set()

            for vr_id, vr_node in self.vertices.items():
                if vr_node.type == 'reticulation' and vr_node.status == UNPROCESSED:
                    if vr_id in processed_vr_ids: 
                        continue

                    structure = self._get_reticulation_structure_vertices(vr_node)
                    if structure:
                        v1, v2, vr, w1, w2, wr = structure
                        if (w1.status == PROCESSED and w2.status == PROCESSED and wr.status == PROCESSED and
                            v1.status == UNPROCESSED and v2.status == UNPROCESSED):
                            ready_retic_structures.append(structure)
                            processed_vr_ids.add(vr_id)

            if ready_retic_structures:
                structure_to_process = ready_retic_structures[0]
                v1, v2, vr, w1, w2, wr = structure_to_process
                print(f"Processing Reticulation Structure centered at {vr.id}")
                if self._process_reticulation_structure(*structure_to_process):
                    progress_made_in_iteration = True
                    processed_nodes_this_run.update([v1.id, v2.id, vr.id])
                    continue

            #Check if finished or stuck
            if not progress_made_in_iteration:
                all_done = all(v.status == PROCESSED for v in self.vertices.values())
                if all_done:
                    print("All nodes processed.")
                    break
                else:
                    #Priority 3: Attempt Non-Time-Consistency Workaround
                    workaround_applied = False
                    for vr_id, vr_node in self.vertices.items():
                        if vr_node.type == 'reticulation' and vr_node.status == UNPROCESSED:
                            structure = self._get_reticulation_structure_vertices(vr_node)
                            if structure:
                                v1, v2, vr, w1, w2, wr = structure
                                if wr.status != PROCESSED: 
                                    continue

                                w1_ready = w1 and w1.status == PROCESSED
                                w2_ready = w2 and w2.status == PROCESSED

                                if w1_ready and not w2_ready and v1.status == UNPROCESSED:
                                    if self._process_reticulation_workaround(vr, v1, w1, wr):
                                        progress_made_in_iteration = True
                                        processed_nodes_this_run.update([v1.id, vr.id])
                                        workaround_applied = True
                                        break
                                    else:
                                        print(f"Workaround failed for {vr.id} via {v1.id}")

                                elif w2_ready and not w1_ready and v2.status == UNPROCESSED:
                                    if self._process_reticulation_workaround(vr, v2, w2, wr):
                                        progress_made_in_iteration = True
                                        processed_nodes_this_run.update([v2.id, vr.id])
                                        workaround_applied = True
                                        break
                                    else:
                                        print(f"Workaround failed for {vr.id} via {v2.id}")

                    if workaround_applied:
                        continue

                    # If no workaround was applied, then truly stuck
                    unprocessed_nodes = [v.id for v in self.vertices.values() if v.status != PROCESSED]
                    print(f"Warning: Processing stuck. No workaround found. Unprocessed: {unprocessed_nodes}")
                    break

            if iteration >= max_iterations:
                print("Warning: Max iterations reached. Possible infinite loop or complex structure.")
                break

        if not all(v.status == PROCESSED for v in self.vertices.values()):
            unprocessed_ids = [v.id for v in self.vertices.values() if v.status != PROCESSED]
            print(f"Warning: Processing finished, but some nodes remain unprocessed: {unprocessed_ids}")

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
                    self._create_visualization_frame("Deleting Reticulation Nodes")
                    break
                else:
                    deleting_retics = False
                    
        while len(self.root.C_prime) > 1:
            self.root.C_prime.pop()
        self.root.final_C = min(self.root.C_prime)

        self._create_visualization_frame("Before Final Cleanup", phase="processing")

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


        # --- Add Frame After Cleanup (with edge categories) ---
        self._create_visualization_frame("After Final Cleanup", phase="final",
                                         scoring_edges=scoring_edges_final,
                                         non_scoring_edges=non_scoring_edges_final)

        # --- Calculate Final Score (using the same logic as categorization) ---
        print("Calculating final score...")
        total_score = len(scoring_edges_final) # The score is simply the count of scoring edges

        print(f"\n--- Final Score: {total_score} ---")

        # --- Add Final Frame with Score (with edge categories) ---
        self._create_visualization_frame(f"Final Score: {total_score}", phase="final",
                                         scoring_edges=scoring_edges_final,
                                         non_scoring_edges=non_scoring_edges_final)

        final_assignment = {vid: v.final_C for vid, v in self.vertices.items()}
        return total_score, final_assignment
    
    def run_approximation(self, char_C, output_video="algo_visualization.gif", fps=1, keep_frames=False):
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
            print(f"Pre-Final Score: {score}")
            self.create_output_video("pre" + output_video, fps, keep_frames)
            #RUN AGAIN
            self._initialize_algorithm(char_C)
            self._iterative_processing()
            score, assignment = self._finalize_assignments_and_score()
            print(f"Final Score: {score}")
            self.create_output_video(output_video, fps, keep_frames) # Create video after scoring
            return score, assignment
        except Exception as e:
            print(f"\n--- Algorithm Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # Optionally try creating video even if scoring failed partially
            self.create_output_video(output_video, fps, keep_frames)
            return None, None