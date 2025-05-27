from latest import PhylogeneticNetwork, PhylogeneticNetwork_Viz, Vertex
from latest.PhylogeneticNetwork import PhylogeneticNetwork
from latest.PhylogeneticNetwork_Viz import PhylogeneticNetwork_Viz
from latest.Vertex import Vertex
from latest.Helper import enewick_from_file, enewick_to_network, read_character_file
from typing import Dict, Tuple, Set, List, Optional
import random
class TestCases:
    def test_case_1(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'leaf')
        wr = Vertex('wr', 'leaf')
        w2 = Vertex('w2', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'w1': 0, 'wr': 0, 'w2': 0}
        return network, char

    def test_case_2(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'leaf')
        wr = Vertex('wr', 'leaf')
        w2 = Vertex('w2', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'w1': 0, 'wr': 0, 'w2': 1}
        return network, char

    def test_case_3_1a(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'tree')
        wr = Vertex('wr', 'leaf')
        w2 = Vertex('w2', 'leaf')
        x1 = Vertex('x1', 'leaf')
        x2 = Vertex('x2', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2, x1, x2]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2'), ('w1', 'x1'), ('w1', 'x2')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'x1': 0, 'x2': 1, 'wr': 0, 'w2': 2}
        return network, char

    def test_case_3_1b(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'tree')
        wr = Vertex('wr', 'leaf')
        w2 = Vertex('w2', 'tree')
        x1 = Vertex('x1', 'leaf')
        x2 = Vertex('x2', 'leaf')
        x3 = Vertex('x3', 'leaf')
        x4 = Vertex('x4', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2, x1, x2, x3, x4]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2'), ('w1', 'x1'), ('w1', 'x2'), ('w2', 'x3'), ('w2', 'x4')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'x1': 0, 'x2': 1, 'wr': 0, 'x3': 0, 'x4': 1}
        return network, char

    def test_case_3_2a(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'leaf')
        wr = Vertex('wr', 'tree')
        w2 = Vertex('w2', 'leaf')
        x1 = Vertex('x1', 'leaf')
        x2 = Vertex('x2', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2, x1, x2]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2'), ('wr', 'x1'), ('wr', 'x2')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'x1': 0, 'x2': 1, 'w1': 0, 'w2': 2}
        return network, char
        
    def test_case_3_2b(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'leaf')
        wr = Vertex('wr', 'tree')
        w2 = Vertex('w2', 'leaf')
        x1 = Vertex('x1', 'leaf')
        x2 = Vertex('x2', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2, x1, x2]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2'), ('wr', 'x1'), ('wr', 'x2')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'x1': 0, 'x2': 1, 'w1': 0, 'w2': 0}
        return network, char

    def test_case_3_3a(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'leaf')
        wr = Vertex('wr', 'leaf')
        w2 = Vertex('w2', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'w1': 0, 'wr': 1, 'w2': 2}
        return network, char
        
    def test_case_3_3b(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        v1 = Vertex('v1', 'tree')
        v2 = Vertex('v2', 'tree')
        vr = Vertex('vr', 'reticulation')
        w1 = Vertex('w1', 'leaf')
        wr = Vertex('wr', 'leaf')
        w2 = Vertex('w2', 'leaf')
        vertices = [root, v1, v2, vr, w1, wr, w2]
        edges = [
            ('root', 'v1'), ('root', 'v2'), ('v1', 'vr'), ('v2', 'vr'),
            ('v1', 'w1'), ('vr', 'wr'), ('v2', 'w2')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'w1': 1, 'wr': 0, 'w2': 1}
        return network, char
            
    def enewick_network1(self, viz: bool = False):
        network = enewick_to_network("(((f,g),(h)#d),(#d,(i,j)))root;", False)
        char={'f': 0, 'g': 1, 'h': 0, 'i': 0, 'j': 1}
        return network, char
        
    def enewick_network2(self, viz: bool = False):
        network = enewick_to_network("((w1,(wr)#vr)v1,(#vr,w2)v2)root;", False)
        char={'w1': 1, 'wr': 0, 'w2': 1}
        return network, char
        
    def handcrafted_network1(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        A = Vertex('A', 'root')
        B = Vertex('B', 'tree')
        C = Vertex('C', 'tree')
        D = Vertex('D', 'tree')
        E = Vertex('E', 'reticulation')
        F = Vertex('F', 'tree')
        G = Vertex('G', 'leaf')
        H = Vertex('H', 'reticulation')
        I = Vertex('I', 'tree')
        J = Vertex('J', 'tree')
        K = Vertex('K', 'leaf')
        L = Vertex('L', 'leaf')
        M = Vertex('M', 'reticulation')
        N = Vertex('N', 'leaf')
        O = Vertex('O', 'leaf')
        network.add_vertex(A)
        network.add_vertex(B)
        network.add_vertex(C)
        network.add_vertex(D)
        network.add_vertex(E)
        network.add_vertex(F)
        network.add_vertex(G)
        network.add_vertex(H)
        network.add_vertex(I)
        network.add_vertex(J)
        network.add_vertex(K)
        network.add_vertex(L)
        network.add_vertex(M)
        network.add_vertex(N)
        network.add_vertex(O)
        network.add_edge('A', 'B')
        network.add_edge('A', 'C')
        network.add_edge('B', 'D')
        network.add_edge('B', 'E')
        network.add_edge('C', 'E')
        network.add_edge('C', 'F')
        network.add_edge('D', 'G')
        network.add_edge('D', 'H')
        network.add_edge('E', 'I')
        network.add_edge('F', 'H')
        network.add_edge('F', 'J')
        network.add_edge('H', 'K')
        network.add_edge('I', 'L')
        network.add_edge('I', 'M')
        network.add_edge('J', 'M')
        network.add_edge('J', 'N')
        network.add_edge('M', 'O')
        char = {'G': 2, 'K': 1, 'L': 0, 'N': 2, 'O': 2}
        return network, char
        
    def handcrafted_network2(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        A = Vertex('A', 'root')
        B = Vertex('B', 'tree')
        C = Vertex('C', 'tree')
        D = Vertex('D', 'tree')
        E = Vertex('E', 'reticulation')
        F = Vertex('F', 'reticulation')
        G = Vertex('G', 'tree')
        H = Vertex('H', 'leaf')
        I = Vertex('I', 'leaf')
        J = Vertex('J', 'leaf')
        K = Vertex('K', 'leaf')
        network.add_vertex(A)
        network.add_vertex(B)
        network.add_vertex(C)
        network.add_vertex(D)
        network.add_vertex(E)
        network.add_vertex(F)
        network.add_vertex(G)
        network.add_vertex(H)
        network.add_vertex(I)
        network.add_vertex(J)
        network.add_vertex(K)
        network.add_edge('A', 'B')
        network.add_edge('A', 'C')
        network.add_edge('B', 'D')
        network.add_edge('B', 'F')
        network.add_edge('C', 'E')
        network.add_edge('C', 'G')
        network.add_edge('D', 'H')
        network.add_edge('D', 'E')
        network.add_edge('E', 'I')
        network.add_edge('F', 'J')
        network.add_edge('G', 'K')
        network.add_edge('G', 'F')
        char = {'H': 0, 'I': 0, 'J': 1, 'K': 1}
        return network, char
        
    def handcrafted_network3(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        A = Vertex('A', 'root')
        B = Vertex('B', 'tree')
        C = Vertex('C', 'tree')
        D = Vertex('D', 'tree')
        E = Vertex('E', 'tree')
        F = Vertex('F', 'tree')
        G = Vertex('G', 'tree')
        H = Vertex('H', 'tree')
        I = Vertex('I', 'reticulation')
        J = Vertex('J', 'tree')
        K = Vertex('K', 'reticulation')
        L = Vertex('L', 'tree')
        M = Vertex('M', 'tree')
        N = Vertex('N', 'leaf')
        O = Vertex('O', 'reticulation')
        P = Vertex('P', 'leaf')
        Q = Vertex('Q', 'leaf')
        R = Vertex('R', 'leaf')
        S = Vertex('S', 'leaf')
        U = Vertex('U', 'reticulation')
        V = Vertex('V', 'leaf')
        W = Vertex('W', 'leaf')
        X = Vertex('X', 'leaf')
        vertices = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, U, V, W, X]
        for vertex in vertices:
            network.add_vertex(vertex)
        edges = [
            ('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('C', 'G'),
            ('D', 'H'), ('D', 'I'), ('E', 'J'), ('E', 'K'), ('F', 'I'), ('F', 'L'),
            ('G', 'K'), ('G', 'M'), ('H', 'N'), ('H', 'O'), ('I', 'P'), ('J', 'O'),
            ('J', 'Q'), ('K', 'R'), ('L', 'S'), ('L', 'U'), ('M', 'U'), ('M', 'V'),
            ('O', 'W'), ('U', 'X')
        ]
        for edge in edges:
            network.add_edge(*edge)
        char = {'N': 0, 'P': 1, 'Q': 2, 'R': 1, 'S': 0, 'V': 0, 'W': 1, 'X': 2}
        return network, char
        
    def handcrafted_network4(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        A = Vertex('A', 'tree')
        B = Vertex('B', 'tree')
        C = Vertex('C', 'tree')
        D = Vertex('D', 'reticulation')
        E = Vertex('E', 'tree')
        F = Vertex('F', 'tree')
        G = Vertex('G', 'reticulation')
        I = Vertex('I', 'reticulation')
        J = Vertex('J', 'tree')
        K = Vertex('K', 'leaf')
        L = Vertex('L', 'leaf')
        M = Vertex('M', 'leaf')
        N = Vertex('N', 'leaf')
        O = Vertex('O', 'leaf')
        vertices = [root, A, B, C, D, E, F, G, I, J, K, L, M, N, O]
        edges = [
            ('root', 'A'), ('root', 'B'), ('A', 'C'), ('A', 'G'), ('B', 'I'), ('B', 'E'), ('C', 'D'), ('C', 'F'), ('D', 'N'),
            ('E', 'D'), ('E', 'J'), ('F', 'K'), ('F', 'I'), ('G', 'M'), ('I', 'O'), ('J', 'L'), ('J', 'G')
        ]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'K': 2, 'L': 1, 'M': 2, 'N': 0, 'O': 1}
        return network, char
        
    def figure_4(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        T1 = Vertex('T1', 'tree')
        T2 = Vertex('T2', 'tree')
        T3 = Vertex('T3', 'tree')
        T4 = Vertex('T4', 'tree')
        T5 = Vertex('T5', 'tree')
        T6 = Vertex('T6', 'tree')
        T7 = Vertex('T7', 'tree')
        T8 = Vertex('T8', 'tree')
        T9 = Vertex('T9', 'tree')
        T10 = Vertex('T10', 'tree')
        T11 = Vertex('T11', 'tree')
        T12 = Vertex('T12', 'tree')
        R1 = Vertex('R1', 'reticulation')
        R2 = Vertex('R2', 'reticulation')
        R3 = Vertex('R3', 'reticulation')
        L1 = Vertex('L1', 'leaf')
        L2 = Vertex('L2', 'leaf')
        L3 = Vertex('L3', 'leaf')
        L4 = Vertex('L4', 'leaf')
        L5 = Vertex('L5', 'leaf')
        L6 = Vertex('L6', 'leaf')
        L7 = Vertex('L7', 'leaf')
        L8 = Vertex('L8', 'leaf')
        L9 = Vertex('L9', 'leaf')
        L10 = Vertex('L10', 'leaf')
        L11 = Vertex('L11', 'leaf')
        vertices = [root, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, R1, R2, R3,
                L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11]
        edges = [('root', 'T1'), ('root', 'T3'), ('T1', 'T2'), ('T1', 'T12'), ('T2', 'T4'), ('T2', 'T10'), ('T3', 'T7'), ('T3', 'L1'), 
                ('T4', 'L2'), ('T4', 'R1'), ('T5', 'R1'), ('T5', 'L4'), ('T6', 'T5'), ('T6', 'L5'), ('T7', 'T6'), ('T7', 'R2'), 
                ('T8', 'L7'), ('T8', 'R2'), ('T9', 'L8'), ('T9', 'T8'), ('T10', 'T9'), ('T10', 'R3'), ('T11', 'R3'), ('T11', 'L10'), 
                ('T12', 'T11'), ('T12', 'L11'), ('R1', 'L3'), ('R2', 'L6'), ('R3', 'L9')]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'L1': 1, 'L2': 1, 'L3': 0, 'L4': 1, 'L5': 0, 'L6': 1, 'L7': 0, 'L8': 1, 'L9': 0, 'L10': 1, 'L11': 0}
        return network, char
        
    def figure_4_OPT(self, viz: bool = False):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        T1 = Vertex('T1', 'tree')
        T2 = Vertex('T2', 'tree')
        T3 = Vertex('T3', 'tree')
        T5 = Vertex('T5', 'tree')
        T6 = Vertex('T6', 'tree')
        T8 = Vertex('T8', 'tree')
        T9 = Vertex('T9', 'tree')
        T11 = Vertex('T11', 'tree')
        T12 = Vertex('T12', 'tree')
        L1 = Vertex('L1', 'leaf')
        L2 = Vertex('L2', 'leaf')
        L3 = Vertex('L3', 'leaf')
        L4 = Vertex('L4', 'leaf')
        L5 = Vertex('L5', 'leaf')
        L6 = Vertex('L6', 'leaf')
        L7 = Vertex('L7', 'leaf')
        L8 = Vertex('L8', 'leaf')
        L9 = Vertex('L9', 'leaf')
        L10 = Vertex('L10', 'leaf')
        L11 = Vertex('L11', 'leaf')
        vertices = [root, T1, T2, T3, T5, T6, T8, T9, T11, T12,
                L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11]
        edges = [('root', 'T1'), ('root', 'T3'), ('T1', 'T2'), ('T1', 'T12'), ('T2', 'L2'), ('T2', 'T9'), ('T3', 'T6'), ('T3', 'L1'), 
                ('T5', 'L3'), ('T5', 'L4'), ('T6', 'T5'), ('T6', 'L5'),  
                ('T8', 'L7'), ('T8', 'L6'), ('T9', 'L8'), ('T9', 'T8'), ('T11', 'L9'), ('T11', 'L10'), 
                ('T12', 'T11'), ('T12', 'L11')]
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        char = {'L1': 1, 'L2': 1, 'L3': 0, 'L4': 1, 'L5': 0, 'L6': 1, 'L7': 0, 'L8': 1, 'L9': 0, 'L10': 1, 'L11': 0}
        return network, char

    def proposition_5(self, viz: bool = False, k: int = 1):
        if viz: network = PhylogeneticNetwork_Viz()
        else: network = PhylogeneticNetwork()
        network.add_vertex(Vertex('root', 'root'))
        char = {}
        for i in range(1, k+1):
            network.add_vertex(Vertex(f'u{i}', 'tree'))
            network.add_vertex(Vertex(f'v{i}', 'tree'))
            network.add_vertex(Vertex(f'w{i}', 'tree'))
            network.add_vertex(Vertex(f'r{i}', 'reticulation'))
            network.add_vertex(Vertex(f'l1{i}', 'leaf'))
            network.add_vertex(Vertex(f'l2{i}', 'leaf'))
            network.add_vertex(Vertex(f'l3{i}', 'leaf'))
            network.add_edge(f'u{i}', f'r{i}')
            network.add_edge(f'v{i}', f'w{i}')
            network.add_edge(f'w{i}', f'r{i}')
            network.add_edge(f'r{i}', f'l1{i}')
            network.add_edge(f'w{i}', f'l2{i}')
            network.add_edge(f'v{i}', f'l3{i}')
            if i == 1:
                network.add_vertex(Vertex('l0', 'leaf'))
                network.add_edge(f'u{i}', 'l0')
                char.update({'l0': 1})
            if (i > 1):
                network.add_edge(f'u{i}', f'v{i-1}')
            if i % 2 == 1:
                char.update({f'l1{i}': 0, f'l2{i}': 1, f'l3{i}': 0})
            else:
                char.update({f'l1{i}': 1, f'l2{i}': 0, f'l3{i}': 1})
            if i%2 == 1:
                network.add_vertex(Vertex(f'T{i}', 'tree'))
                if i == 1:
                    network.add_edge('root', f'T{i}')
                else:
                    network.add_edge(f'T{i-2}', f'T{i}')
            else:
                network.add_vertex(Vertex(f'T{i}', 'tree'))
                if i == 2:
                    network.add_edge('root', f'T{i}')
                else:
                    network.add_edge(f'T{i-2}', f'T{i}')
            network.add_edge(f'T{i}', f'u{i}')
            if i == k:
                network.add_edge(f'T{k}', f'v{k}')
        network.add_vertex(Vertex(f'u{k+1}', 'leaf'))
        if k == 1:
            network.add_edge('root', f'u{k+1}')
        else:
            network.add_edge(f'T{k-1}', f'u{k+1}')
        char.update({f'u{k+1}': k%2})
        return network, char
    
    def random_network(self, input_size: int = 1, rate1: float = 0.5, rate2: float = 0.5, rate3: float = 0.5):
        def add_tree(network, parent, i):
            tmp_vertex = Vertex(f'{i}', 'tree')
            network.add_vertex(tmp_vertex)
            network.add_edge(parent.id, f'{i}')
            retic_parents.append(tmp_vertex)
            if len(parent.children) == 2:
                retic_parents.remove(parent)
                
        def add_tree1(network, parent, i):
            tmp_vertex = Vertex(f'{i}', 'tree')
            network.add_vertex(tmp_vertex)
            network.add_edge(parent.id, f'{i}')
            retic_parents.append(tmp_vertex)
            if parent.type == 'reticulation':
                non_retic_parents.remove(parent)
            if len(parent.children) == 2:
                non_retic_parents.remove(parent)
                
        def add_reticulation(network, parent, i):
            retic_parents.remove(parent)
            second_parent = random.choice(retic_parents)
            retic_parents.append(parent)
            while (len(parent.children) > 0 and second_parent == parent.children[0]) or (len(second_parent.children) > 0 and parent == second_parent.children[0]):
                parent = random.choice(retic_parents)
                retic_parents.remove(parent)
                second_parent = random.choice(retic_parents)
                retic_parents.append(parent)
            if parent == None:
                print(retic_parents)
            tmp_vertex = Vertex(f'{i}', 'reticulation')
            network.add_vertex(tmp_vertex)
            network.add_edge(parent.id, f'{i}')
            retic_parents.remove(parent)
            if len(parent.children) == 1:
                non_retic_parents.append(parent)
            network.add_edge(second_parent.id, f'{i}')
            retic_parents.remove(second_parent)
            if len(second_parent.children) == 1:
                non_retic_parents.append(second_parent)
            non_retic_parents.append(tmp_vertex)
        def not_related(parent1, parent2):
            if parent1 == None or parent2 == None:
                return False
            if len(parent1.children) > 0 and len(parent2.children) > 0:
                if parent1 == parent2.children[0] or parent2 == parent1.children[0]:
                    return False
            if len(parent1.children) > 0 and len(parent2.children) == 0:
                if parent1.children[0] == parent2:
                    return False
            if len(parent1.children) == 0 and len(parent2.children) > 0:
                if parent2.children[0] == parent1:
                    return False
            return True
           
        network = PhylogeneticNetwork()
        root = Vertex('root', 'root')
        root_viz = Vertex('root', 'root')
        network.add_vertex(root)
        char = {}
        non_retic_parents = [root]
        retic_parents = []
        for i in range(1, input_size+1):
            if len(retic_parents) > 2 or (len(retic_parents) == 2 and not_related(retic_parents[0], retic_parents[1])):
                if len(non_retic_parents) > 0:
                    if random.random() < rate1: #high rate1 means we prefer to process reticulation now rather than later.
                        parent = random.choice(retic_parents)
                        if random.random() < rate2: #high rate2 means high chance we fill potential reticulation slot with tree node
                            add_tree(network, parent, i)
                        else:
                            add_reticulation(network, parent, i)
                    else:
                        parent = random.choice(non_retic_parents)
                        add_tree1(network, parent, i)
                else:
                    parent = random.choice(retic_parents)
                    if random.random() < rate3: #high rate3 means high chance we fill potential reticulation slot with tree node
                        add_tree(network, parent, i)
                    else:
                        add_reticulation(network, parent, i)
            else:
                if len(non_retic_parents) > 0:
                    parent = random.choice(non_retic_parents)
                    add_tree1(network, parent, i)
                else:
                    parent = random.choice(retic_parents)
                    add_tree(network, parent, i)
                
        count = 1
        retic_count = 0
        tree_count = -1
        vertices = []
        edges = []
        for v in network.vertices.values():
            if v.type in ['root', 'tree']:
                tree_count += 1
                length = len(v.children)
                while length < 2:
                    vertices.append(Vertex(f'l{count}', 'leaf'))
                    edges.append((v.id, f'l{count}'))
                    char.update({f'l{count}': random.randint(0, 1)})
                    count += 1
                    length += 1
            elif v.type == 'reticulation':
                retic_count += 1
                if len(v.children) < 1:
                    vertices.append(Vertex(f'l{count}', 'leaf'))
                    edges.append((v.id, f'l{count}'))
                    char.update({f'l{count}': random.randint(0, 1)})
                    count += 1
        for v in vertices:
            network.add_vertex(v)
        for e in edges:
            network.add_edge(*e)
        return network, char, (retic_count/(retic_count+tree_count))
    

    def random_network_new(
        input_size: int = 1,
        rate1: float = 0.5,
        rate2: float = 0.5,
        rate3: float = 0.5,
        seed: Optional[int] = None,
        verify: bool = True,
    ) -> Tuple[PhylogeneticNetwork_Viz, PhylogeneticNetwork, Dict[str, int], float]:
        """Generate a binary, *tree-child* phylogenetic network.

        Parameters
        ----------
        input_size:
            Number of internal vertices (tree or reticulation) that will be inserted
            **before** the terminal leaves are attached.  Values in the range 20–200
            create moderately sized examples.
        rate1, rate2, rate3:
            Probabilities that bias *when* and *how* reticulations are created.
            They roughly correspond to the original paper's idea but can be tuned
            without breaking correctness:

            * ``rate1`` - probability that we prioritise *closing* an open
            reticulation over simply extending the tree when both choices are
            available.
            * ``rate2`` - probability that, given such a reticulation parent, we
            still decide to attach a *tree* child instead of a reticulation child
            (=> delays the actual reticulation event).
            * ``rate3`` - same as ``rate2`` but for the situation where
            *only* reticulation parents exist.
        seed:
            RNG seed for reproducibility.  Pass ``None`` (default) for non‑deterministic
            output.
        verify:
            If *True* (default) a lightweight verifier runs at the end and raises
            ``ValueError`` if the resulting graph is *not* a binary tree‑child DAG.

        Returns
        -------
        network_viz, network, char_vector, retic_ratio
            ``network_viz`` is usually identical to ``network`` but keeps the door
            open for richer visualisation back‑ends.  ``char_vector`` is a random
            0/1 assignment on the leaves; ``retic_ratio`` equals
            ``#reticulation_vertices / (#reticulation_vertices + #tree_vertices)``.
        """

        # ---------------------------------------------------------------------
        #  House‑keeping & bookkeeping structures
        # ---------------------------------------------------------------------
        if seed is not None:
            random.seed(seed)

        net_viz = PhylogeneticNetwork_Viz()
        net = PhylogeneticNetwork()

        root = Vertex("root", "root")
        net.add_vertex(root)
        net_viz.add_vertex(root)

        # Vertices that *can still accept another child* -----------------------
        # ``tree_parents``   – any vertex that can accept a *tree* child.
        # ``retic_parents``  – tree vertices that can still act as *a parent of a
        #                      reticulation vertex* (i.e. out‑deg < 2).
        tree_parents: Set[Vertex] = {root}
        retic_parents: Set[Vertex] = {root}

        # Statistics -----------------------------------------------------------
        tree_internal = 0  # *excluding* the root
        retic_internal = 0

        # ---------------------------------------------------------------------
        #  Helper closures
        # ---------------------------------------------------------------------
        def _update_parent_sets(v: Vertex) -> None:  # noqa: D401, D413
            """(Re‑)enter or leave *v* in the helper sets according to its degree."""
            if v.type != "reticulation":
                if len(v.children) < 2:
                    tree_parents.add(v)
                    retic_parents.add(v)
                else:
                    tree_parents.discard(v)
                    retic_parents.discard(v)
            else:  # reticulation
                # Reticulation may *receive* a tree child but may never be parent
                # of another reticulation.
                if len(v.children) < 1:
                    tree_parents.add(v)
                else:
                    tree_parents.discard(v)

        def _new_tree(parent: Vertex, idx: int) -> Vertex:  # noqa: D401, D413
            nonlocal tree_internal
            v = Vertex(f"t{idx}", "tree")
            net.add_vertex(v)
            net.add_edge(parent.id, v.id)
            net_viz.add_vertex(v)
            net_viz.add_edge(parent.id, v.id)
            tree_internal += 1
            _update_parent_sets(parent)
            _update_parent_sets(v)
            return v

        def _new_reticulation(parent: Vertex, idx: int) -> Vertex:  # noqa: D401, D413
            nonlocal retic_internal

            # Choose the *second* parent first so we do not accidentally pick the
            # same vertex twice after we attach the new node.
            # ────────────────────────────────────────────────────────────────────
            second_parent_pool = retic_parents - {parent}
            if not second_parent_pool:
                # Fallback – treat like a tree event instead of raising.
                return _new_tree(parent, idx)
            second_parent = random.choice(list(second_parent_pool))

            v = Vertex(f"r{idx}", "reticulation")
            net.add_vertex(v)
            net_viz.add_vertex(v)

            # First edge
            net.add_edge(parent.id, v.id)
            net_viz.add_edge(parent.id, v.id)

            # Second edge
            net.add_edge(second_parent.id, v.id)
            net_viz.add_edge(second_parent.id, v.id)

            retic_internal += 1
            # Update helper sets ------------------------------------------------
            _update_parent_sets(parent)
            _update_parent_sets(second_parent)
            _update_parent_sets(v)
            return v

        # ---------------------------------------------------------------------
        #  Phase 1 – create *input_size* internal vertices
        # ---------------------------------------------------------------------
        for idx in range(1, input_size + 1):
            # Decision tree copied from the original algorithm, cleaned up for sets
            if len(retic_parents) > 1:  # We *can* create a reticulation
                if len(tree_parents) > 1:
                    if random.random() < rate1:
                        parent = random.choice(list(retic_parents))
                        if random.random() < rate2:
                            _new_tree(parent, idx)
                        else:
                            _new_reticulation(parent, idx)
                    else:
                        parent = random.choice(list(tree_parents))
                        _new_tree(parent, idx)
                else:  # Only reticulation parents left
                    parent = random.choice(list(retic_parents))
                    if random.random() < rate3:
                        _new_tree(parent, idx)
                    else:
                        _new_reticulation(parent, idx)
            else:  # We *cannot* create a reticulation right now
                parent = random.choice(list(tree_parents))
                _new_tree(parent, idx)

        # ---------------------------------------------------------------------
        #  Phase 2 – attach leaves to satisfy binary, tree‑child conditions
        # ---------------------------------------------------------------------
        leaf_counter = 1
        char_vec: Dict[str, int] = {}

        def _attach_leaf(parent: Vertex) -> None:  # noqa: D401, D413
            nonlocal leaf_counter
            leaf = Vertex(f"l{leaf_counter}", "leaf")
            leaf_counter += 1
            char_vec[leaf.id] = random.randint(0, 1)
            net.add_vertex(leaf)
            net_viz.add_vertex(leaf)
            net.add_edge(parent.id, leaf.id)
            net_viz.add_edge(parent.id, leaf.id)

        for v in list(net.vertices.values()):  # snapshot, we will add to net
            if v.type in {"root", "tree"}:
                while len(v.children) < 2:
                    _attach_leaf(v)
            elif v.type == "reticulation" and len(v.children) < 1:
                _attach_leaf(v)

        # Extra tree‑child safeguard – make sure every *tree* vertex has at least
        # one *non‑retic* child
        for v in list(net.vertices.values()):
            if v.type in {"root", "tree"} and all(c.type == "reticulation" for c in v.children):
                _attach_leaf(v)

        # ---------------------------------------------------------------------
        #  Optional: verify integrity
        # ---------------------------------------------------------------------
        if verify:
            _verify_tree_child_binarity(net)

        # Ratio (root is *not* counted as internal tree vertex here) ------------
        retic_ratio = retic_internal / (retic_internal + tree_internal) if (retic_internal + tree_internal) else 0.0

        return net_viz, net, char_vec, retic_ratio


# ---------------------------------------------------------------------------
#  Mini verifier (linear‑time)
# ---------------------------------------------------------------------------

def _verify_tree_child_binarity(net: PhylogeneticNetwork) -> None:  # noqa: D401, D413
    """Raise ``ValueError`` if *net* is not a binary tree‑child DAG."""

    for v in net.vertices.values():
        indeg = len(v.parents)
        outdeg = len(v.children)

        if v.type == "root":
            if indeg != 0 or outdeg != 2:
                raise ValueError(f"Root degree violation at {v.id}")
        elif v.type == "leaf":
            if outdeg != 0 or indeg < 1:
                raise ValueError(f"Leaf degree violation at {v.id}")
        elif v.type == "tree":
            if indeg != 1 or outdeg not in {1, 2}:
                raise ValueError(f"Tree vertex degree violation at {v.id}")
            if all(c.type == "reticulation" for c in v.children):
                raise ValueError(f"Tree‑child property violated at {v.id}")
        elif v.type == "reticulation":
            if indeg != 2 or outdeg != 1:
                raise ValueError(f"Reticulation degree violation at {v.id}")
        else:
            raise ValueError(f"Unknown vertex type '{v.type}' at {v.id}")
