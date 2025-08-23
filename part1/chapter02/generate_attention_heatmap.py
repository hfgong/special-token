#!/usr/bin/env python3
"""
Generate attention score heatmap for SEP token visualization
This replaces the complex TikZ figure with a simpler Python-generated version
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

def generate_attention_heatmap():
    """Generate attention score heatmap showing SEP token patterns"""
    
    # Define tokens
    tokens = ['[CLS]', 'The', 'cat', 'sleeps', '[SEP]', 'A', 'feline', 'rests', '[SEP]']
    n_tokens = len(tokens)
    
    # Create attention pattern matrix
    # Initialize with random baseline attention
    np.random.seed(42)
    attention_matrix = np.random.rand(n_tokens, n_tokens) * 0.3
    
    # Add self-attention (diagonal)
    np.fill_diagonal(attention_matrix, 0.8)
    
    # Add strong attention to CLS token (first column)
    attention_matrix[:, 0] += 0.3
    
    # Add SEP token patterns (indices 4 and 8)
    sep1_idx = 4
    sep2_idx = 8
    
    # SEP tokens attend strongly to segment boundaries
    attention_matrix[sep1_idx, :] = 0.2  # Reset SEP1 row
    attention_matrix[sep1_idx, 0:5] = 0.6  # SEP1 attends to first segment
    attention_matrix[sep1_idx, sep1_idx] = 0.9  # Self-attention
    
    attention_matrix[sep2_idx, :] = 0.2  # Reset SEP2 row
    attention_matrix[sep2_idx, 5:] = 0.6  # SEP2 attends to second segment
    attention_matrix[sep2_idx, sep2_idx] = 0.9  # Self-attention
    
    # Other tokens attend to SEP tokens moderately
    attention_matrix[:, sep1_idx] += 0.2
    attention_matrix[:, sep2_idx] += 0.2
    
    # Cross-segment attention (weaker)
    attention_matrix[1:4, 5:8] = 0.15  # First segment to second
    attention_matrix[5:8, 1:4] = 0.15  # Second segment to first
    
    # Normalize to [0, 1]
    attention_matrix = np.clip(attention_matrix, 0, 1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Attention Heatmap
    im = ax1.imshow(attention_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax1.set_xticks(range(n_tokens))
    ax1.set_yticks(range(n_tokens))
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(tokens, fontsize=10)
    
    # Add labels
    ax1.set_xlabel('Key Tokens', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Query Tokens', fontsize=11, fontweight='bold')
    ax1.set_title('Attention Score Matrix with SEP Tokens', fontsize=12, fontweight='bold')
    
    # Highlight SEP token rows and columns
    for sep_idx in [sep1_idx, sep2_idx]:
        # Highlight row
        rect = Rectangle((-0.5, sep_idx - 0.5), n_tokens, 1, 
                        linewidth=2, edgecolor='purple', facecolor='none')
        ax1.add_patch(rect)
        # Highlight column
        rect = Rectangle((sep_idx - 0.5, -0.5), 1, n_tokens,
                        linewidth=2, edgecolor='purple', facecolor='none')
        ax1.add_patch(rect)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Score', rotation=270, labelpad=15, fontsize=10)
    
    # Plot 2: Attention Pattern Visualization
    ax2.axis('off')
    
    # Token positions for visualization
    y_pos = 0.7
    x_positions = np.linspace(0.1, 0.9, n_tokens)
    
    # Draw tokens
    for i, (x, token) in enumerate(zip(x_positions, tokens)):
        if token == '[CLS]':
            color = '#FBB405'  # Orange
        elif token in ['[SEP]']:
            color = '#8E24F5'  # Purple
        else:
            color = '#E8F0FE'  # Light blue
        
        # Token box
        rect = mpatches.FancyBboxPatch((x-0.04, y_pos-0.03), 0.08, 0.06,
                                      boxstyle="round,pad=0.01",
                                      facecolor=color, edgecolor='black',
                                      linewidth=1.5 if '[' in token else 1)
        ax2.add_patch(rect)
        ax2.text(x, y_pos, token, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw segment indicators
    ax2.plot([x_positions[0], x_positions[3]], [y_pos-0.08, y_pos-0.08], 
            'b-', linewidth=3, label='Segment A')
    ax2.plot([x_positions[5], x_positions[7]], [y_pos-0.08, y_pos-0.08], 
            'g-', linewidth=3, label='Segment B')
    
    # Add attention flow arrows (simplified)
    # Within-segment attention
    for i in range(1, 4):
        for j in range(1, 4):
            if i != j:
                ax2.annotate('', xy=(x_positions[j], y_pos-0.04),
                           xytext=(x_positions[i], y_pos-0.04),
                           arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3, color='blue'))
    
    for i in range(5, 8):
        for j in range(5, 8):
            if i != j:
                ax2.annotate('', xy=(x_positions[j], y_pos-0.04),
                           xytext=(x_positions[i], y_pos-0.04),
                           arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3, color='green'))
    
    # SEP-mediated cross-segment attention
    ax2.annotate('', xy=(x_positions[6], y_pos+0.15),
               xytext=(x_positions[2], y_pos+0.15),
               arrowprops=dict(arrowstyle='<->', lw=2, alpha=0.7, color='purple',
                             connectionstyle="arc3,rad=0.3"))
    
    # Labels
    ax2.text(0.5, 0.9, 'Attention Flow Patterns', ha='center', fontsize=12, fontweight='bold')
    ax2.text(0.5, 0.45, 'Key Patterns:', ha='center', fontsize=11, fontweight='bold')
    
    # Pattern descriptions
    patterns = [
        ('Within-Segment:', 'Local context & syntactic relations', 'blue'),
        ('SEP-Mediated:', 'Cross-segment bridge & boundary info', 'purple'),
        ('Cross-Segment:', 'Semantic alignment & entailment', 'red')
    ]
    
    y_text = 0.35
    for pattern, desc, color in patterns:
        ax2.text(0.2, y_text, pattern, fontsize=10, fontweight='bold', color=color)
        ax2.text(0.35, y_text, desc, fontsize=9)
        y_text -= 0.08
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def save_figure():
    """Generate and save the attention heatmap figure"""
    fig = generate_attention_heatmap()
    
    # Save in multiple formats
    fig.savefig('fig_sep_attention_flow.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('fig_sep_attention_flow.png', dpi=300, bbox_inches='tight')
    
    print("Attention heatmap saved as fig_sep_attention_flow.pdf and .png")
    
    # Show the figure (optional)
    plt.show()

if __name__ == "__main__":
    save_figure()