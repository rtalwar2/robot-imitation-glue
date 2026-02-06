import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    # 1. Setup the Canvas
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')  # Hide the axes

    # --- STYLE CONFIGURATION ---
    # Colors aligned with your charts:
    # Grey (Vision), Red/Pink (Audio), Green (Oracle/Wire), Purple (Expert/Policy)
    colors = {
        'vision': '#A9A9A9',    # Grey
        'audio': '#FF6B6B',     # Soft Red
        'oracle': '#77DD77',    # Pastel Green
        'policy': '#B39EB5',    # Pastel Purple
        'action': '#333333'     # Dark Grey
    }
    
    # Box styles
    box_props = dict(boxstyle="round,pad=0.6", ec="none", alpha=0.9)
    dashed_box_props = dict(boxstyle="round,pad=0.6", ec=colors['oracle'], 
                           fc="white", lw=2, linestyle="--", alpha=1.0)

    # --- HELPER FUNCTION TO DRAW BOXES ---
    def add_node(x, y, text, color, subtext=None, props=box_props, text_color="white"):
        # Draw the box
        if props == dashed_box_props:
            # Special handling for the empty dashed box
            ax.add_patch(patches.FancyBboxPatch((x-1.5, y-0.75), 3, 1.5, 
                                               boxstyle="round,pad=0.2", 
                                               ec=colors['oracle'], fc="white", 
                                               linestyle="--", lw=2))
            t_color = colors['oracle'] # Text color for the dashed box
        else:
            ax.text(x, y, text, ha='center', va='center', size=14, weight='bold', 
                    color=text_color, bbox=dict(facecolor=color, **props))
            t_color = text_color
            
        # Add subtext (smaller description below main text)
        if subtext:
            ax.text(x, y-0.35, subtext, ha='center', va='center', size=9, 
                    color='black', style='italic')

    # --- DRAWING NODES ---

    # COLUMN 1: SENSORS (Inputs)
    add_node(2, 6.5, "RGB Camera", colors['vision'], "Standard Input")
    add_node(2, 4.0, "Microphone", colors['audio'], "Gripper Mounted")
    add_node(2, 1.5, "Microcontroller", colors['oracle'], "Privileged Info\n(Wire Signal)", props=dashed_box_props)

    # COLUMN 2: ENCODERS (Processing)
    add_node(6, 6.5, "Vision Encoder", colors['vision'], "ResNet / ViT")
    add_node(6, 4.0, "Audio Spectrum\nTransformer (AST)", colors['audio'], "Learns the 'Click'")
    
    # The Oracle Logic
    ax.text(6, 1.5, "Ground Truth\nState Detection", ha='center', va='center', 
            size=10, color=colors['oracle'], weight='bold',
            bbox=dict(facecolor='white', edgecolor=colors['oracle'], boxstyle='round,pad=0.5', linestyle='--'))

    # COLUMN 3: POLICY (The Brain)
    # Drawing a big box for the Diffusion Policy
    ax.add_patch(patches.FancyBboxPatch((8.5, 2.5), 3, 5, boxstyle="round,pad=0.4", 
                                       facecolor=colors['policy'], alpha=0.2))
    ax.text(10, 7.8, "Diffusion Policy", ha='center', size=16, weight='bold', color='#555')
    
    # Inside the Policy
    ax.text(10, 5, "Feature\nFusion", ha='center', va='center', size=12, 
            bbox=dict(facecolor='white', boxstyle='circle,pad=0.5', ec='gray'))

    # COLUMN 4: OUTPUT
    add_node(13, 5, "Robot Action", colors['action'], "Velocity / Pose")

    # --- DRAWING ARROWS ---
    arrow_props = dict(arrowstyle="->", color="#555", lw=2, mutation_scale=20)
    dashed_arrow = dict(arrowstyle="->", color=colors['oracle'], lw=2, mutation_scale=20, linestyle="--")

    # Connect Vision
    ax.annotate("", xy=(4.8, 6.5), xytext=(3.2, 6.5), arrowprops=arrow_props) # Cam -> Enc
    ax.annotate("", xy=(9.2, 5.2), xytext=(7.2, 6.5), arrowprops=arrow_props) # Enc -> Fusion

    # Connect Audio
    ax.annotate("", xy=(4.5, 4.0), xytext=(3.0, 4.0), arrowprops=arrow_props) # Mic -> AST
    ax.annotate("", xy=(9.2, 5.0), xytext=(7.5, 4.0), arrowprops=arrow_props) # AST -> Fusion

    # Connect Oracle (Dashed to indicate training only)
    ax.annotate("", xy=(4.7, 1.5), xytext=(3.5, 1.5), arrowprops=dashed_arrow) # Wire -> Logic
    ax.annotate("", xy=(9.0, 4.5), xytext=(7.3, 1.5), arrowprops=dashed_arrow) # Logic -> Fusion
    
    # Connect Policy to Output
    ax.annotate("", xy=(12.2, 5), xytext=(11.5, 5), arrowprops=arrow_props)

    # --- ANNOTATIONS / STORYTELLING ---
    
    # Label: Privileged Information
    ax.text(4.5, 0.8, "Training Only\n(Privileged Info)", ha='center', color=colors['oracle'], weight='bold')
    
    # Label: Transfer
    ax.text(6, 2.8, "↑\nAudio learns to\nreplace this", ha='center', size=9, color='#555')

    plt.tight_layout()
    plt.show()

# Run the drawing function
draw_architecture()