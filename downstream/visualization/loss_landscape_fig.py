import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F
from hxe import HierarchicalCrossEntropyLoss
from create_tree import all_tokens, get_classes, get_weighting

# Setup
vocab_size = 70
tree = get_classes(all_tokens)[0]
weights = get_weighting(all_tokens, "exponential", value=0.2)
classes = get_classes(all_tokens)[0]
hxe = HierarchicalCrossEntropyLoss(all_tokens, classes, weights)

def cross_entropy_loss(true_probs, pred_probs):
    return F.cross_entropy(torch.tensor(pred_probs), torch.tensor(true_probs).argmax(dim=0)).item()

def hierarchical_cross_entropy_loss(true_probs, pred_probs):
    return hxe(torch.tensor(pred_probs).unsqueeze(0), torch.tensor(true_probs).argmax(dim=0)).item()

# Create different predicted probability distributions
pred_probs_list = [
    np.full((vocab_size, vocab_size), 1/vocab_size),  # Uniform distribution
    np.random.dirichlet(np.ones(vocab_size), size=vocab_size),  # Random distribution
    np.clip(np.eye(vocab_size) + 0.1 * np.random.randn(vocab_size, vocab_size), 0, 1)  # Near-perfect predictions
]
titles = ['Uniform Distribution', 'Random Distribution', 'Near-Perfect Predictions']

fig = plt.figure(figsize=(20, 20))
fig.suptitle('Comparison of Cross-Entropy and Hierarchical Cross-Entropy Loss', fontsize=16)

for i, (pred_probs, title) in enumerate(zip(pred_probs_list, titles)):
    # Normalize predictions
    pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
    
    # Calculate losses
    ce_losses = np.array([[cross_entropy_loss(np.eye(vocab_size)[j], pred_probs[k]) 
                           for k in range(vocab_size)] for j in range(vocab_size)])
    hce_losses = np.array([[hierarchical_cross_entropy_loss(np.eye(vocab_size)[j], pred_probs[k]) 
                            for k in range(vocab_size)] for j in range(vocab_size)])
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(range(vocab_size), range(vocab_size))
    
    # Cross-Entropy Loss
    ax1 = fig.add_subplot(3, 2, 2*i+1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, ce_losses, cmap='viridis', edgecolor='none')
    ax1.set_title(f'Cross-Entropy Loss\n{title}')
    ax1.set_xlabel('Predicted Token')
    ax1.set_ylabel('True Token')
    ax1.set_zlabel('Loss')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Hierarchical Cross-Entropy Loss
    ax2 = fig.add_subplot(3, 2, 2*i+2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, hce_losses, cmap='viridis', edgecolor='none')
    ax2.set_title(f'Hierarchical Cross-Entropy Loss\n{title}')
    ax2.set_xlabel('Predicted Token')
    ax2.set_ylabel('True Token')
    ax2.set_zlabel('Loss')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('3d_loss_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('3d_loss_comparison.pdf', bbox_inches='tight')
plt.savefig('3d_loss_comparison.png', bbox_inches='tight')