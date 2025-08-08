
import numpy as np
from math import exp, fsum
from nltk.tree import Tree
from copy import deepcopy

from hxe import get_label

v_codon = Tree("V", ["GTT", "GTC", "GTA", "GTG"])
a_codon = Tree("A", ["GCT", "GCC", "GCA", "GCG"])
d_codon = Tree("D", ["GAT", "GAC"])
e_codon = Tree("E", ["GAA", "GAG"])
g_codon = Tree("G", ["GGT", "GGC", "GGA", "GGG"])
f_codon = Tree("F", ["TTT", "TTC"])
l_codon = Tree("L", ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"])
s_codon = Tree("S", ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"])
y_codon = Tree("Y", ["TAT", "TAC"])
c_codon = Tree("C", ["TGT", "TGC"])
w_codon = Tree("W", ["TGG"])
p_codon = Tree("P", ["CCT", "CCC", "CCA", "CCG"])
h_codon = Tree("H", ["CAT", "CAC"])
q_codon = Tree("Q", ["CAA", "CAG"])
r_codon = Tree("R", ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"])
i_codon = Tree("I", ["ATT", "ATC", "ATA"])
t_codon = Tree("T", ["ACT", "ACC", "ACA", "ACG"])
n_codon = Tree("N", ["AAT", "AAC"])
k_codon = Tree("K", ["AAA", "AAG"])

start_codon = Tree("start_codon", ["ATG"])
end_codon = Tree("end_codon", ["TAA", "TAG", "TGA"])

aa_codons = Tree("aa_codons", [v_codon, a_codon, d_codon, e_codon, g_codon, f_codon, l_codon, 
                               s_codon, y_codon, c_codon, w_codon, p_codon, h_codon, q_codon, 
                               r_codon, i_codon, t_codon, n_codon, k_codon])

codons_tree = Tree("codons", [start_codon, aa_codons, end_codon])
special_tokens = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<distill_token>"]
all_tokens = Tree("all_tokens", [codons_tree] + [Tree(t, [t]) for t in special_tokens])

def get_uniform_weighting(hierarchy: Tree, value):
    """
    Construct unit weighting tree from hierarchy.

    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The value to fill the tree with.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    for p in weights.treepositions():
        node = weights[p]
        if isinstance(node, Tree):
            node.set_label(value)
        else:
            weights[p] = value
    return weights


def get_exponential_weighting(hierarchy: Tree, value, normalize=True):
    """
    Construct exponentially decreasing weighting, where each edge is weighted
    according to its distance from the root as exp(-value*dist).

    Args:
        hierarchy: The hierarchy to use to generate the weights.
        value: The decay value.
        normalize: If True ensures that the sum of all weights sums
            to one.

    Returns:
        Weights as a nltk.Tree whose labels are the weights associated with the
        parent edge.
    """
    weights = deepcopy(hierarchy)
    all_weights = []
    for p in weights.treepositions():
        node = weights[p]
        weight = exp(-value * len(p))
        all_weights.append(weight)
        if isinstance(node, Tree):
            node.set_label(weight)
        else:
            weights[p] = weight
    total = fsum(all_weights)  # stable sum
    if normalize:
        for p in weights.treepositions():
            node = weights[p]
            if isinstance(node, Tree):
                node.set_label(node.label() / total)
            else:
                weights[p] /= total
    return weights


def get_weighting(hierarchy: Tree, weighting="uniform", **kwargs):
    """
    Get different weightings of edges in a tree.

    Args:
        hierarchy: The tree to generate the weighting for.
        weighting: The type of weighting, one of 'uniform', 'exponential'.
        **kwards: Keyword arguments passed to the weighting function.
    """
    if weighting == "uniform":
        return get_uniform_weighting(hierarchy, **kwargs)
    elif weighting == "exponential":
        return get_exponential_weighting(hierarchy, **kwargs)
    else:
        raise NotImplementedError("Weighting {} is not implemented".format(weighting))
    
def get_classes(hierarchy: Tree, output_all_nodes=False):
    """
    Return all classes associated with a hierarchy. The classes are sorted in
    alphabetical order using their label, putting all leaf nodes first and the
    non-leaf nodes afterwards.

    Args:
        hierarhcy: The hierarchy to use.
        all_nodes: Set to true if the non-leaf nodes (excepted the origin) must
            also be included.

    Return:
        A pair (classes, positions) of the array of all classes (sorted) and the
        associated tree positions.
    """

    def get_classes_from_positions(positions):
        classes = [get_label(hierarchy[p]) for p in positions]
        class_order = np.argsort(classes)  # we output classes in alphabetical order
        positions = [positions[i] for i in class_order]
        classes = [classes[i] for i in class_order]
        return classes, positions

    positions = hierarchy.treepositions("leaves")
    classes, positions = get_classes_from_positions(positions)

    if output_all_nodes:
        positions_nl = [p for p in hierarchy.treepositions() if p not in positions]
        classes_nl, positions_nl = get_classes_from_positions(positions_nl)
        classes += classes_nl
        positions += positions_nl

    return classes, positions

import random
from collections import defaultdict

def collect_leaves(tree):
    leaves = []
    for subtree in tree.subtrees():
        if isinstance(subtree, str) or len(subtree) == 0:
            leaves.append(subtree)
    return leaves

def shuffle_and_merge_codons(tree):
    if tree.label() != 'all_tokens':
        return tree

    new_tree = Tree('all_tokens', [])
    codons_tree = Tree('codons', [])
    new_tree.append(codons_tree)

    # Collect all codon leaves
    codon_leaves = []
    for subtree in tree.subtrees():
        if subtree.label() in ['start_codon', 'end_codon']:
            codon_leaves.extend(subtree.leaves())
        elif subtree.label() in list('ACDEFGHIKLMNPQRSTVWY'):
            codon_leaves.extend([leaf for leaf in subtree.leaves() if isinstance(leaf, str) and len(leaf) == 3 and leaf.isalpha()])

    # Shuffle all codon leaves
    random.shuffle(codon_leaves)

    # Distribute shuffled codons to amino acid groups
    aa_groups = []
    i = 0
    while i < len(codon_leaves):
        group_size = random.randint(1, min(6, len(codon_leaves) - i))
        aa_groups.append(codon_leaves[i:i+group_size])
        i += group_size

    # Create initial amino acid subtrees
    initial_groups = {}
    for group in aa_groups:
        if 'ATG' in group:
            label = 'start_codon'
        elif set(group) & {'TAA', 'TAG', 'TGA'}:
            label = 'end_codon'
        else:
            label = random.choice(list('ACDEFGHIKLMNPQRSTVWY'))
        
        if label in initial_groups:
            initial_groups[label].extend(group)
        else:
            initial_groups[label] = group

    # Merge groups with the same label and assign random names
    merged_groups = []
    used_numbers = set()
    for label, codons in initial_groups.items():
        if label in ['start_codon', 'end_codon']:
            merged_groups.append((label, codons))
        else:
            while True:
                random_number = random.randint(1, 100)
                if random_number not in used_numbers:
                    used_numbers.add(random_number)
                    new_label = f"random#{random_number}"
                    merged_groups.append((new_label, codons))
                    break

    # Add merged groups to the codons tree
    for label, codons in merged_groups:
        codons_tree.append(Tree(label, codons))

    # Add special tokens
    for subtree in tree:
        if isinstance(subtree, Tree) and subtree.label() not in ['codons', 'aa_codons']:
            new_tree.append(subtree)

    return new_tree

