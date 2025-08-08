import torch
import torch.nn as nn
from nltk.tree import Tree
import numpy as np

class NLTKHierarchicalSoftmax(nn.Module):
    def __init__(self, nhid, nltk_tree):
        super(NLTKHierarchicalSoftmax, self).__init__()
        self.nhid = nhid
        self.tree = nltk_tree
        self.node_params = nn.ParameterDict()
        self.leaf_to_path = {}
        
        self._initialize_parameters(self.tree)
        self._create_leaf_paths(self.tree)

    def _initialize_parameters(self, tree, path='0'):
        if not isinstance(tree, Tree):
            return  # Leaf node
        
        num_children = len(tree)
        param_name = path.replace('.', '_')
        self.node_params[param_name] = nn.Parameter(torch.randn(self.nhid, num_children))
        
        for i, subtree in enumerate(tree):
            self._initialize_parameters(subtree, f"{path}.{i}")

    def _create_leaf_paths(self, tree, path='0'):
        if not isinstance(tree, Tree):
            self.leaf_to_path[tree] = path
            return
        
        for i, subtree in enumerate(tree):
            self._create_leaf_paths(subtree, f"{path}.{i}")

    def forward(self, inputs, labels=None):
        batch_size = inputs.size(0)
        
        if labels is not None:
            # Compute probabilities only for the paths to the given labels
            probs = torch.ones(batch_size, device=inputs.device)
            for i in range(batch_size):
                path = self.leaf_to_path[labels[i]]
                node_path = '0'
                for step in path.split('.')[1:]:
                    param_name = node_path.replace('.', '_')
                    logits = torch.matmul(inputs[i], self.node_params[param_name])
                    probs[i] *= torch.softmax(logits, dim=0)[int(step)]
                    node_path = f"{node_path}.{step}"
            return probs
        else:
            # Compute probabilities for all leaves
            all_leaves = list(self.leaf_to_path.keys())
            probs = torch.zeros(batch_size, len(all_leaves), device=inputs.device)
            for i, leaf in enumerate(all_leaves):
                path = self.leaf_to_path[leaf]
                leaf_prob = torch.ones(batch_size, device=inputs.device)
                node_path = '0'
                for step in path.split('.')[1:]:
                    param_name = node_path.replace('.', '_')
                    logits = torch.matmul(inputs, self.node_params[param_name])
                    leaf_prob *= torch.softmax(logits, dim=1)[:, int(step)]
                    node_path = f"{node_path}.{step}"
                probs[:, i] = leaf_prob
            return probs

# Example usage:
# tree = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
# hsoftmax = NLTKHierarchicalSoftmax(hidden_dim, tree)
# output = hsoftmax(input_tensor, labels)