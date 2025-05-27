import collections
import re
import pathlib
import time
from typing import Dict, Set
from latest import MPNet, PhylogeneticNetwork, PhylogeneticNetwork_Viz, Vertex
from PhylogeneticNetwork import PhylogeneticNetwork
from PhylogeneticNetwork_Viz import PhylogeneticNetwork_Viz
from Vertex import Vertex
from MPNet import MPNet

def parse_enewick(s: str) -> tuple[set[str], set[tuple[str, str]]]:
    """
    Parse an (extended) Newick string and return its vertices and directed edges.
    Duplicate labels e.g. '#d' are treated as the same vertex (reticulation).
    """
    def norm(label: str) -> str:
        """Return the canonical vertex name (strip exactly one leading #)."""
        return label[1:] if label.startswith('#') else label

    label = re.compile(r"[A-Za-z0-9#]+")
    MARK   = object()                         # unique stack sentinel

    stack:  list = []
    edges:  set[tuple[str, str]] = set()
    verts:  set[str] = set()

    i = 0
    while i < len(s):
        char = s[i]

        if char == '(':
            stack.append(MARK)

        elif char == ')':
            # --- pop all children until '('
            children = []
            while stack and stack[-1] is not MARK:
                children.append(stack.pop())
            if not stack:
                raise ValueError("unmatched ')'")
            stack.pop()                      # discard MARK

            # --- optional label immediately after ')'
            m = label.match(s, i + 1)
            if m:
                parent = norm(m.group())
                verts.add(parent)
                i = m.end() - 1              # advance past label
            else:
                parent = f'X{i}'             # anonymous internal node
                verts.add(parent)

            # --- connect every child to the parent
            for c in children:
                edges.add((parent, c))
            stack.append(parent)

        elif char == ',' or char in '; \t\n\r':
            pass                             # delimiters / whitespace

        else:                                # bare label
            m = label.match(s, i)
            v = norm(m.group())
            verts.add(v)
            stack.append(v)
            i = m.end() - 1                  # jump to last char of label

        i += 1

    if len(stack) != 1:
        raise ValueError("malformed Newick string")

    root = stack.pop().rstrip(';')           # final item is the root
    verts.add(root)
    return verts, edges

def enewick_to_network(s: str) -> PhylogeneticNetwork:
    network = PhylogeneticNetwork()
    
    vertices, edges = parse_enewick(s)
    print(f"Parsed vertices: {vertices}")
    print(f"Parsed edges: {edges}")
    incoming_edges = collections.Counter()
    outgoing_edges = collections.Counter()
    for edge in edges:
        outgoing_edges[edge[0]] += 1
        incoming_edges[edge[1]] += 1

    for vertex in vertices:
        if vertex not in incoming_edges:
            network.add_vertex(Vertex(vertex, 'root'))
        elif vertex not in outgoing_edges:
            network.add_vertex(Vertex(vertex, 'leaf'))
        elif incoming_edges[vertex] == 1 and outgoing_edges[vertex] == 2:
            network.add_vertex(Vertex(vertex, 'tree'))
        elif incoming_edges[vertex] == 2 and outgoing_edges[vertex] == 1:
            network.add_vertex(Vertex(vertex, 'reticulation'))
        else:
            raise ValueError(f"Vertex {vertex} has an invalid number of incoming and outgoing edges")
    
    for edge in edges:
        u, v = edge
        if u not in network.vertices or v not in network.vertices:
            raise ValueError(f"Edge {edge} contains invalid vertices")
        network.add_edge(u, v)
    
    return network

def enewick_to_network(s: str, viz: bool = False) -> PhylogeneticNetwork:
    if viz: network = PhylogeneticNetwork_Viz()
    else: network = PhylogeneticNetwork()
    
    vertices, edges = parse_enewick(s)
    incoming_edges = collections.Counter()
    outgoing_edges = collections.Counter()
    for edge in edges:
        outgoing_edges[edge[0]] += 1
        incoming_edges[edge[1]] += 1

    for vertex in vertices:
        if vertex not in incoming_edges:
            network.add_vertex(Vertex(vertex, 'root'))
        elif vertex not in outgoing_edges:
            network.add_vertex(Vertex(vertex, 'leaf'))
        elif incoming_edges[vertex] == 1 and outgoing_edges[vertex] == 2:
            network.add_vertex(Vertex(vertex, 'tree'))
        elif incoming_edges[vertex] == 2 and outgoing_edges[vertex] == 1:
            network.add_vertex(Vertex(vertex, 'reticulation'))
        else:
            raise ValueError(f"Vertex {vertex} has an invalid number of incoming and outgoing edges")
    
    for edge in edges:
        u, v = edge
        if u not in network.vertices or v not in network.vertices:
            raise ValueError(f"Edge {edge} contains invalid vertices")
        network.add_edge(u, v)
    
    return network

def read_character_file(path: str) -> dict[str, str]:
    char_C = {}
    with open(path) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            vid, state, *_ = ln.split()
            char_C[vid] = try_convert_to_int(state)
    return char_C

def try_convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        return s

def enewick_from_file(path: str) -> str:
    with open(path) as fh:
        return fh.read().strip()
    
def run_from_files(network_txt: str, character_txt: str, viz: bool = False, output_video: str = None):
    nwk      = enewick_from_file(network_txt)
    network  = enewick_to_network(nwk, viz)
    char_C   = read_character_file(character_txt)
    if viz: network.run_approximation(char_C=char_C, output_video=output_video)
    else: network.run_approximation(char_C)
    


# ───────────────────────────────────────────────────────────────────────────
#  1.  Canonicalise reticulation labels and strip internal names  (same logic
#      as in the earlier message, but wrapped so we can call it here)
# ───────────────────────────────────────────────────────────────────────────
import re

def _normalise_enewick(raw_newick: str) -> str:
    """Return an MPNet-ready e-Newick string (internal names removed,
       reticulations renamed #H<n>)."""
    # --- remove internal tree-node names, keep root & #H.. -----------
    m_root = re.search(r'\)([^):;,]+)\s*;', raw_newick)
    root_label = m_root.group(1) if m_root else None
    pat_internal = re.compile(r'\)([A-Za-z0-9_]+)')
    produced = set()

    def _kill(m: re.Match[str]) -> str:
        lab = m.group(1)
        if lab == root_label or lab.startswith('#'):
            return ')' + lab
        return ')'
    stripped = pat_internal.sub(_kill, raw_newick)

    # --- convert any non-standard retic label → #H<n> -----------------
    pat_retic = re.compile(r'#([A-Za-z0-9_]+)')
    mapping: Dict[str, str] = {}

    def _to_Hn(m: re.Match[str]) -> str:
        token = m.group(0)
        if re.fullmatch(r'#H\d+', token):
            return token
        if token not in mapping:
            mapping[token] = f'#H{len(mapping)+1}'
        return mapping[token]

    return pat_retic.sub(_to_Hn, stripped)


# ───────────────────────────────────────────────────────────────────────────
#  2.  Build an extended-Newick string from the PhylogeneticNetwork object
# ───────────────────────────────────────────────────────────────────────────
def _build_raw_enewick_recursive(net: "PhylogeneticNetwork") -> str:
    """Serialise *net* to an extended-Newick string in which every vertex has
       its current id as label.  Branch lengths/support values are ignored."""
    # 2.1  give every reticulation vertex a canonical '#H<n>' label
    retic_map: Dict[int, str] = {}      # vertex-id -> #Hn
    next_n = 1
    for v in net.vertices.values():
        if v.type == "reticulation":
            if isinstance(v.id, str) and re.fullmatch(r'#H\d+', v.id):
                retic_map[v.id] = v.id         # already canonical
            else:
                retic_map[v.id] = f"#H{next_n}"
                next_n += 1

    produced_full: Set[int] = set()     # reticulation vertices whose full
                                        # "(child)#Hn" form has been output

    def encode(v) -> str:
        if v.type == "leaf":
            return str(v.id)

        if v.type == "reticulation":
            label = retic_map[v.id]
            if v.id not in produced_full:          # first time → full form
                produced_full.add(v.id)
                child_txt = encode(v.children[0])
                return f"({child_txt}){label}"
            return label                           # subsequent → just #Hn

        # tree or root
        child_txts = [encode(c) for c in v.children]
        inside = ",".join(child_txts)
        if v.type == "root":
            return f"({inside}){v.id}"
        return f"({inside})"                       # internal tree node

    return encode(net.root) + ";"                 # terminate the tree

import re
from typing import Dict, Set, List, Tuple

def _build_raw_enewick(net: "PhylogeneticNetwork") -> str:
    """
    Serialise *net* to an extended-Newick string without using recursion.
    The output is byte-for-byte identical to the original recursive
    `_build_raw_enewick`:

    • Every reticulation vertex is given a canonical '#H<n>' label.
    • The *first* time a reticulation appears it is emitted as
      '(child)#Hn'; later appearances are just '#Hn'.
    • Internal tree vertices are printed without a label, the root keeps
      its id, and leaves keep their ids.
    """
    # ------------------------------------------------------------------
    # 1. Canonical labels for reticulations
    retic_map: Dict[int, str] = {}
    next_n = 1
    for v in net.vertices.values():
        if v.type == "reticulation":
            if isinstance(v.id, str) and re.fullmatch(r"#H\d+", v.id):
                retic_map[v.id] = v.id          # already canonical
            else:
                retic_map[v.id] = f"#H{next_n}"
                next_n += 1

    produced_full: Set[int] = set()             # which retics got "(child)#Hn"
    output_map: Dict[int, str] = {}             # vertex-id → serialised text
    stack: List[Tuple["Vertex", bool]] = [(net.root, False)]
    #                    └─ visited flag: False = pre-order, True = post-order

    while stack:
        node, visited = stack.pop()

        # -------- leaves ------------------------------------------------
        if node.type == "leaf":
            output_map[node.id] = str(node.id)
            continue

        # -------- reticulation vertices ---------------------------------
        if node.type == "reticulation":
            label = retic_map[node.id]

            if visited:                         # post-order → emit "(child)#Hn"
                child_txt = output_map[node.children[0].id]
                output_map[node.id] = f"({child_txt}){label}"
            elif node.id not in produced_full:  # first encounter
                produced_full.add(node.id)
                stack.append((node, True))      # come back after the child
                stack.append((node.children[0], False))
            else:                               # subsequent encounter
                output_map[node.id] = label
            continue

        # -------- internal tree vertex or root --------------------------
        if visited:                             # post-order → we have children
            child_txts = [output_map[c.id] for c in node.children]
            inside = ",".join(child_txts)
            if node.type == "root":
                output_map[node.id] = f"({inside}){node.id}"
            else:
                output_map[node.id] = f"({inside})"
        else:                                   # pre-order → push children
            stack.append((node, True))
            for child in reversed(node.children):   # preserve left-to-right
                stack.append((child, False))

    return output_map[net.root.id] + ";"



# ───────────────────────────────────────────────────────────────────────────
#  3.  Write both files
# ───────────────────────────────────────────────────────────────────────────
def export_mpnet_files(network: "PhylogeneticNetwork",
                       newick_path: str | pathlib.Path = "network.txt",
                       character_path: str | pathlib.Path = "character.txt",
                       sort_leaves: bool = True) -> None:
    """
    Create the two text files needed by MPNet from *network*.

    Parameters
    ----------
    network : PhylogeneticNetwork
        The network whose leaves already carry their states
        (`leaf.input_C` or `leaf.state`).
    newick_path : str | Path
        Output path for the e-Newick file (default "network.txt").
    character_path : str | Path
        Output path for the character file (default "character.txt").
    sort_leaves : bool
        If True (default) the character lines are written in
        lexicographic leaf-id order so the file is deterministic.
    """
    # 3.1  produce & normalise the e-Newick
    raw   = _build_raw_enewick(network)
    clean = _normalise_enewick(raw)
    pathlib.Path(newick_path).write_text(clean + "\n")

    # 3.2  character file
    leaf_pairs = []
    for leaf in network.leaves:
        if hasattr(leaf, "input_C"):
            state = leaf.input_C
        elif hasattr(leaf, "state"):
            state = leaf.state
        else:
            raise ValueError(f"Leaf {leaf.id} carries no character state.")
        leaf_pairs.append((str(leaf.id), str(state)))

    if sort_leaves:
        leaf_pairs.sort(key=lambda p: p[0])

    lines = [f"{leaf} {state}" for leaf, state in leaf_pairs]
    pathlib.Path(character_path).write_text("\n".join(lines) + "\n")

def compare_to_exact(network: PhylogeneticNetwork, char: dict[str, str], network_str: str, character_str: str):
    network._initialize_algorithm(char)
    folder = "C:\\Users\\jantj\\Documents\\mpnet\\"
    export_mpnet_files(network, folder+network_str, folder+character_str)
    start = time.perf_counter()
    exact_score = MPNet(folder+network_str, folder+character_str)
    exact_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    pre_score, score, _ = network.run_approximation(char)
    algo_elapsed = time.perf_counter() - start
    pre_ratio  = pre_score  / exact_score if exact_score else None
    if pre_ratio is None or pre_ratio <= 2:
        pathlib.Path(folder+network_str).unlink()
        pathlib.Path(folder+character_str).unlink()
    return (pre_score, score, exact_score, algo_elapsed, exact_elapsed)

# vertices = 7
# id = 9538
# run_from_files(f"C:\\Users\\jantj\\Documents\\mpnet\\random_network_{vertices}_{id}__network.txt", f"C:\\Users\\jantj\\Documents\\mpnet\\random_network_{vertices}_{id}__character.txt", True, f"{vertices}_{id}.gif")
# print(MPNet(f"C:\\Users\\jantj\\Documents\\mpnet\\random_network_{vertices}_{id}_network.txt", f"C:\\Users\\jantj\\Documents\\mpnet\\random_network_{vertices}_{id}_character.txt"))