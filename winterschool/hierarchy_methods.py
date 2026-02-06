import matplotlib.pyplot as plt

def draw_methods_hierarchy():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # --- STYLES ---
    bbox_style = dict(boxstyle="round,pad=0.5", fc="white", ec="#333", lw=2)
    arrow_props = dict(arrowstyle="->", color="#555", lw=2, mutation_scale=15)
    
    # Colors for differentiation
    c_soft = '#FFD700'  # Gold for Soft Sensor
    c_embed = '#87CEEB' # SkyBlue for Embeddings
    c_freeze = '#D3D3D3' # Grey for Frozen
    c_unfreeze = '#FF6B6B' # Red for Unfrozen

    # --- HELPER ---
    def add_box(x, y, text, color="white", subtext=None, width=None):
        ax.text(x, y, text, ha='center', va='center', size=11, weight='bold',
                bbox=dict(boxstyle=f"round,pad=0.5", fc=color, ec="#333", lw=1.5))
        if subtext:
            ax.text(x, y-0.4, subtext, ha='center', va='top', size=8, style='italic', color="#444")

    # --- NODES ---

    # Root
    add_box(6, 7.5, "Audio Input\n(Microphone)", "#eee")

    # Level 1: The Split
    add_box(3, 5.5, "Strategy A:\nSoft Sensor", c_soft, "Swapping Wire for Audio")
    add_box(9, 5.5, "Strategy B:\nDeep Fusion", c_embed, "Audio inside Policy")

    # Level 2: Deep Fusion details
    add_box(7, 3.5, "Classification\nHead", "white", "Probability Score")
    add_box(11, 3.5, "AST\nEmbeddings", "white", "Rich Feature Vector")

    # Level 3: Frozen vs Unfrozen (The hierarchy)
    # Grouping them to save space
    ax.text(9, 1.5, "Training Dynamics", ha='center', weight='bold', size=10)
    
    # Drawing the "Switch" for training
    ax.add_patch(plt.Rectangle((6.5, 0.5), 5, 1.2, fill=True, color="#f9f9f9", ec="#ddd"))
    
    add_box(7.5, 1.1, "Frozen AST", c_freeze, "Pre-trained only")
    add_box(10.5, 1.1, "Unfrozen AST", c_unfreeze, "Finetuned E2E")

    # --- CONNECTIONS ---
    
    # Root to Split
    ax.annotate("", xy=(3, 6.0), xytext=(6, 7.0), arrowprops=arrow_props)
    ax.annotate("", xy=(9, 6.0), xytext=(6, 7.0), arrowprops=arrow_props)

    # Strategy B split
    ax.annotate("", xy=(7, 4.0), xytext=(9, 5.0), arrowprops=arrow_props)
    ax.annotate("", xy=(11, 4.0), xytext=(9, 5.0), arrowprops=arrow_props)

    # Connections to Training Dynamics
    ax.annotate("", xy=(9, 1.8), xytext=(7, 3.0), arrowprops=dict(arrowstyle="->", color="#555", lw=1, ls="--"))
    ax.annotate("", xy=(9, 1.8), xytext=(11, 3.0), arrowprops=dict(arrowstyle="->", color="#555", lw=1, ls="--"))

    # Annotations
    ax.text(3, 4.5, "Output matches\nWire Signal (0/1)", ha='center', size=9, color="#555")
    ax.text(9, 2.5, "Can apply to both inputs:", ha='center', size=9, color="pink")

    plt.tight_layout()
    plt.show()

draw_methods_hierarchy()