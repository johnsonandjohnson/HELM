import pickle
import random
import torch
from nltk import Tree
from create_tree import all_tokens, get_classes
import lightning as L
import torch
from utils import AA_TO_ID
from losses import DistillationLoss
import numpy as np
from itertools import chain

tree = all_tokens

class DistanceDict(dict):
    """
    Small helper class implementing a symmetrical dictionary to hold distance data.
    """

    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}

    def __getitem__(self, i):
        if i[0] == i[1]:
            return 0
        else:
            return self.distances[(i[0], i[1]) if i[0] < i[1] else (i[1], i[0])]

    def __setitem__(self, i):
        raise NotImplementedError()



def build_parent_dict(hierarchy):
    parent_dict = {}
    for parent, children in hierarchy.items():
        for child in children:
            parent_dict[child] = parent
    return parent_dict

def get_path_to_root(node, parent_dict):
    path = []
    while node in parent_dict:
        path.append(node)
        node = parent_dict[node]
    path.append(node)  # Append the root
    return path[::-1]

def find_lca(node1, node2, parent_dict):
    path1 = get_path_to_root(node1, parent_dict)
    path2 = get_path_to_root(node2, parent_dict)
    lca = None
    for ancestor1, ancestor2 in zip(path1, path2):
        if ancestor1 == ancestor2:
            lca = ancestor1
        else:
            break
    return lca

def height_of_node(node, parent_dict):
    height = 0
    while node in parent_dict:
        height += 1
        node = parent_dict[node]
    return height

def h_distance(leaf1, leaf2, hierarchy):
    parent_dict = build_parent_dict(hierarchy)
    if leaf1 == leaf2:
        return 0
    else:
        return height_of_node(find_lca(leaf1, leaf2, parent_dict), parent_dict)

def make_all_soft_labels(distances, classes, hardness):
    distance_matrix = torch.Tensor([[distances[c1, c2] for c1 in classes] for c2 in classes])
    max_distance = torch.max(distance_matrix)
    distance_matrix /= max_distance
    soft_labels = torch.exp(-hardness * distance_matrix) / torch.sum(torch.exp(-hardness * distance_matrix), dim=0)
    return soft_labels

def make_batch_soft_labels(all_soft_labels, target, num_classes, batch_size, device):
    soft_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32).to(device)
    for i in range(batch_size):
        this_label = all_soft_labels[:, target[i]]
        soft_labels[i, :] = this_label
    return soft_labels

hierarchy = {}
for subtree in tree.subtrees():
    hierarchy[subtree.label()] = []
    for child in subtree:
        if isinstance(child, Tree):
            hierarchy[subtree.label()].append(child.label())
        else:
            hierarchy[subtree.label()].append(child)
height = tree.height() - 1
all_nodes = set(list(hierarchy.keys()) + list(chain(*list(hierarchy.values()))))
distance_dict = {}
for node in all_nodes:
    for other_node in all_nodes:
        distance_dict[(node, other_node)] = height - h_distance(node, other_node, hierarchy)

classes = get_classes(tree)[0]
soft_labels = make_all_soft_labels(distance_dict, classes, 5)
