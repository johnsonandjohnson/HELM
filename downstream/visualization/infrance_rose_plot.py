import pickle
import torch
from utils import ID_TO_CODON
from create_tree import get_classes, all_tokens
import matplotlib.pyplot as plt
import numpy as np
from Bio.Seq import Seq
import colorsys
from collections import defaultdict
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors

classes = get_classes(all_tokens)[0]

def codon_to_aa(codon):
    try:
        if str(Seq(codon).translate()) == "*":
            return "Stop"
        return str(Seq(codon).translate())
    except:
        return codon


with open("./xe.pkl", "rb") as f:
    output_xe , target_xe  = pickle.load(f)

with open("./hxe0.2.pkl", "rb") as f:
    output_hxe2 , target_hxe2  = pickle.load(f)

with open("./hxe0.4.pkl", "rb") as f:
    output_hxe4 , target_hxe4  = pickle.load(f)

with open("./hxe0.6.pkl", "rb") as f:
    output_hxe6 , target_hxe6  = pickle.load(f)

print("".join([ID_TO_CODON[i] for i in target_xe[146]]))
max_diff_all = -1000

output_xe = torch.tensor(output_xe).softmax(dim=-1)
output_hxe2 = torch.tensor(output_hxe2).softmax(dim=-1)
output_hxe4 = torch.tensor(output_hxe4).softmax(dim=-1)
output_hxe6 = torch.tensor(output_hxe6).softmax(dim=-1)
keys_xe = list(ID_TO_CODON.values())
keys_hxe = classes

for i in [146]:
    for j in [31]:
        true_codon = ID_TO_CODON[target_xe[i, j].item()]
        true_str = ""
        for item in target_xe[i, :]:
            true_str += ID_TO_CODON[item.item()]

        xe_probs = {k.replace("T", "U"): v.item() for k, v in zip(keys_xe, output_xe[i, j, :])}
        hxe2_probs = {k.replace("T", "U"): v.item() for k, v in zip(keys_hxe, output_hxe2[i, j, :])}
        hxe4_probs = {k.replace("T", "U"): v.item() for k, v in zip(keys_hxe, output_hxe4[i, j, :])}
        hxe6_probs = {k.replace("T", "U"): v.item() for k, v in zip(keys_hxe, output_hxe6[i, j, :])}
    
        xe_aa_probs = defaultdict(float)
        hxe2_aa_probs = defaultdict(float)
        hxe4_aa_probs = defaultdict(float)
        hxe6_aa_probs = defaultdict(float)

        for codon, prob in xe_probs.items():
            aa = codon_to_aa(codon)
            xe_aa_probs[aa] += prob

        for codon, prob in hxe2_probs.items():
            aa = codon_to_aa(codon)
            hxe2_aa_probs[aa] += prob

        for codon, prob in hxe4_probs.items():
            aa = codon_to_aa(codon)
            hxe4_aa_probs[aa] += prob

        for codon, prob in hxe6_probs.items():
            aa = codon_to_aa(codon)
            hxe6_aa_probs[aa] += prob

        xe_hxe2_diff = {aa: hxe2_aa_probs[aa] - xe_aa_probs[aa] for aa in set(xe_aa_probs.keys())}[codon_to_aa(true_codon)]
        xe_hxe4_diff = {aa: hxe4_aa_probs[aa] - xe_aa_probs[aa] for aa in set(xe_aa_probs.keys())}[codon_to_aa(true_codon)]
        xe_hxe6_diff = {aa: hxe6_aa_probs[aa] - xe_aa_probs[aa] for aa in set(xe_aa_probs.keys())}[codon_to_aa(true_codon)]
        max_diff = max(xe_hxe2_diff, xe_hxe4_diff, xe_hxe6_diff)
        xe_probs_all = all([p < 0.5 for p in xe_probs.values()])
        if max_diff == xe_hxe2_diff:
            best_hxe = hxe2_probs
            best_output = output_hxe2
            best_aa_probs = hxe2_aa_probs
        elif max_diff == xe_hxe4_diff:
            best_hxe = hxe4_probs
            best_output = output_hxe4
            best_aa_probs = hxe4_aa_probs
        else:
            best_hxe = hxe6_probs
            best_output = output_hxe6
            best_aa_probs = hxe6_aa_probs
        hxe_probs_all = all([p < 0.5 for p in best_hxe.values()])
        if max_diff_all < max_diff and (ID_TO_CODON[output_xe[i, j, :].argmax(-1).item()] != ID_TO_CODON[target_xe[i, j].item()]) and (classes[best_output[i, j, :].argmax(-1).item()] != classes[target_hxe2[i, j].item()]) and xe_probs_all and hxe_probs_all and best_aa_probs[codon_to_aa(true_codon)] > 0.9 and xe_aa_probs[codon_to_aa(true_codon)] > 0.15:
            max_diff_all = max_diff
            print(max_diff_all)
            string_true = true_str
            pos_true = j
            best_i = i
            max_value_xe = xe_probs
            max_value_hxe = best_hxe
            true_codon = ID_TO_CODON[target_xe[i, j].item()]
        if max_diff_all > 0.7:
            break
    if max_diff_all > 0.7:
        break


def normalize_probs(xe_probs, hxe_probs):
    all_probs = list(xe_probs.values()) + list(hxe_probs.values())
    max_prob = max(all_probs)
    return {k: v / max_prob for k, v in xe_probs.items()}, {k: v / max_prob for k, v in hxe_probs.items()}

def group_codons_by_aa(codons):
    aa_to_codons = defaultdict(list)
    for codon in codons:
        aa = codon_to_aa(codon)
        if aa != "<pad>" and aa != "<eos>" and aa != "<cls>" and aa != "<pad>" and aa != "<unk>" and aa != "<mask>" and aa != "<distill_token>":
            aa_to_codons[aa].append(codon)
    return aa_to_codons

def create_rose_plot(xe_probs, hxe_probs, title, true_codon):
    codons = list(xe_probs.keys())
    codons = sorted(codons, reverse=True)
    
    # Group codons by amino acid
    aa_to_codons = group_codons_by_aa(codons)
    # Create a new ordered list of codons
    aa_l = sorted(aa_to_codons.keys(), key=lambda x: len(aa_to_codons[x]), reverse=True)
    print(aa_l)

    ordered_codons = [codon for aa in aa_l for codon in aa_to_codons[aa]]
    print(ordered_codons)
    n = len(ordered_codons)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(projection='polar'))
    
    width = (2 * np.pi) / n
    
    # Use colorblind-friendly colors
    color_xe = mcolors.CSS4_COLORS['blue']
    color_hxe = mcolors.CSS4_COLORS['orange']
    
    # Plot XE probabilities
    xe_bars = ax.bar(theta, [xe_probs[codon] for codon in ordered_codons],
                     width=width, bottom=0.0, color=color_xe, alpha=0.7, label='XE')
    
    # Plot HXE probabilities
    hxe_bars = ax.bar(theta, [hxe_probs[codon] for codon in ordered_codons],
                      width=width, bottom=0.0, color=color_hxe, alpha=0.7, label='HELM')

    # Outer circle (amino acids) and separation lines
    for aa, aa_codons in aa_to_codons.items():
        if aa not in ["<pad>", "<eos>", "<cls>", "<unk>", "<mask>", "<distill_token>"]:
            start_angle = theta[ordered_codons.index(aa_codons[0])]
            end_angle = theta[ordered_codons.index(aa_codons[-1])] + width
            mid_angle = (start_angle + end_angle) / 2

            # Draw bold separation line
            # ax.plot([start_angle, start_angle], [0, 1], color='brown', linewidth=2.5, linestyle='-.')

            # Add amino acid label further outside the plot area
            label_style = 'bold' if aa == codon_to_aa(true_codon) else 'normal'
            label_color = 'green' if aa == codon_to_aa(true_codon) else 'black'
            # ax.text(mid_angle, 3, aa, ha='center', va='center', fontsize=22, fontweight=label_style,
            #         color=label_color, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Set codon labels
    # ax.set_xticks(theta)
    # ax.set_xticklabels(ordered_codons, fontsize=18)

     # Add extra padding to x-tick labels
    # for label in ax.get_xticklabels():
    #     label.set_y(-0.3)  # Adjust this value to increase the distance from the plot

    
    # # Highlight and annotate the true codon
    # for i, codon in enumerate(ordered_codons):
    #     if codon == true_codon:
    #         max_prob = max(xe_probs[codon], hxe_probs[codon])
    #         ax.text(theta[i], max_prob + 0.1, 'True Codon', 
    #                 ha='center', va='bottom', fontweight='bold', color='green', fontsize=14,
    #                 bbox=dict(facecolor='white', edgecolor='green', alpha=0.7))
    #         ax.plot([theta[i], theta[i]], [0, 1], 
    #                 color='green', linewidth=3, linestyle='-')
    #         # ax.get_xticklabels()[i].set_fontweight('bold')
    #         # ax.get_xticklabels()[i].set_color('green')
    
    # Add probability labels
    for i, codon in enumerate(ordered_codons):
        xe_prob = xe_probs[codon]
        hxe_prob = hxe_probs[codon]
        max_prob = max(xe_prob, hxe_prob)
        # max_prob = hxe_prob
        
        # if max_prob > 0.01:  # Only show labels for probabilities > 1%
        #     ax.text(theta[i], max_prob, f'{max_prob:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold', pad=1)
    
    # Use log scale for radial axis to show small probabilities
    ax.set_yscale('symlog', linthresh=0.01)
    
    # Adjust plot limits
    ax.set_ylim(0, 1)  # Set radial limit to 1
    
    # Remove radial ticks and labels
    ax.set_yticks([])
      # Remove all ticks and labels
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])  # This removes the theta labels
    
    
    # Set title with proper positioning
    # ax.set_title(title, fontsize=24, y=1.1)
    
    # Make the legend larger and move it outside the plot
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=18)

    plt.tight_layout()
    # legend_properties = {'weight':'bold', 'size': 28}
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, prop=legend_properties)
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()



# Create plots
create_rose_plot(max_value_xe, max_value_hxe, "combined_rose", true_codon.replace("T", "U"))
