import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge
import colorsys

import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
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


def create_codon_wheel(output_file='codon_wheel.png', dpi=300, figsize=(16, 16)):
    codon_table = {
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': 'Stop', 'UAG': 'Stop',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'UGU': 'C', 'UGC': 'C', 'UGA': 'Stop', 'UGG': 'W',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    # Group codons by amino acid
    aa_codons = {}
    for codon, aa in codon_table.items():
        if aa not in aa_codons:
            aa_codons[aa] = []
        aa_codons[aa].append(codon)
    # Create a list of amino acids sorted by number of codons (descending)
    sorted_aa = sorted(aa_codons.keys(), key=lambda x: len(aa_codons[x]), reverse=True)
    amino_acid_colors = {}
    for i, aa in enumerate(sorted_aa):
        hue = i / len(sorted_aa)
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        amino_acid_colors[aa] = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    outer_radius = 12
    inner_radius = 10
    text_radius = outer_radius - 0.7

    def polar_to_cartesian(r, theta):
        return r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))

    def draw_separation_line(angle):
        x_start, y_start = polar_to_cartesian(0, angle)
        x_end, y_end = polar_to_cartesian(outer_radius + 0.5, angle)
        ax.plot([x_start, x_end], [y_start, y_end], color='black', linewidth=1)

    start_angle = 0
    for aa in sorted_aa:
        codons = aa_codons[aa]
        angle_per_codon = 360 / 64  # Total angle divided by total number of codons
        end_angle = start_angle + angle_per_codon * len(codons)
        
        # Draw the wedge for this amino acid
        wedge = Wedge((0, 0), outer_radius, start_angle, end_angle,
                      width=outer_radius-inner_radius, 
                      facecolor=amino_acid_colors[aa],
                      edgecolor='white', linewidth=0.5)
        ax.add_patch(wedge)
        
        # Add codons
        for i, codon in enumerate(codons):
            codon_start = start_angle + i * angle_per_codon
            codon_end = codon_start + angle_per_codon
            codon_mid = (codon_start + codon_end) / 2
            x, y = polar_to_cartesian(text_radius, codon_mid)
            ax.text(x, y, codon, ha='center', va='center', fontsize=16, fontweight='bold',
                    rotation=codon_mid-90 if codon_mid <= 180 else codon_mid+90)
        
        # Add amino acid label
        aa_angle = (start_angle + end_angle) / 2
        x, y = polar_to_cartesian(inner_radius - 0.5, aa_angle)
        ax.text(x, y, aa, ha='center', va='center', fontsize=24, fontweight='bold',
                bbox=dict(facecolor="white", edgecolor='none', alpha=0.6, pad=0.5))
        
        # Draw separation line
        draw_separation_line(end_angle)
        
        start_angle = end_angle

    # Draw the inner circle
    inner_circle = Circle((0, 0), inner_radius - 1, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(inner_circle)

    ax.set_xlim(-outer_radius-1, outer_radius+1)
    ax.set_ylim(-outer_radius-1, outer_radius+1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close()

# Generate the wheel
create_codon_wheel()