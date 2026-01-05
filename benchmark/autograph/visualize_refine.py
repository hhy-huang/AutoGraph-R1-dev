import json
import networkx as nx
import matplotlib.pyplot as plt

def load_refinement_result(json_path):
    """Load refinement result from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def subgraph_to_networkx(subgraph):
    """Convert subgraph (list of dicts with subject, relation, object) to NetworkX DiGraph"""
    G = nx.DiGraph()
    
    for triple in subgraph:
        subject = triple.get("subject", "")
        relation = triple.get("relation", "")
        obj = triple.get("object", "")
        
        if subject and obj:
            # Add nodes
            if not G.has_node(subject):
                G.add_node(subject, label=subject)
            if not G.has_node(obj):
                G.add_node(obj, label=obj)
            
            # Add edge with relation as label
            G.add_edge(subject, obj, relation=relation, label=relation)
    
    return G

def truncate_label(label, max_length=40):
    """Truncate long labels with ellipsis"""
    if len(label) > max_length:
        return label[:max_length-3] + "..."
    return label

def visualize_subgraph(G, title, pos=None, figsize=(20, 12), node_size=2000, font_size=8, ax=None):
    """Visualize a NetworkX graph"""
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        plt.sca(ax)
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "Empty Graph", ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        return pos
    
    # Use spring layout if position not provided
    if pos is None:
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=node_size,
                          alpha=0.9,
                          ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           edge_color='gray',
                           arrows=True,
                           arrowsize=20,
                           alpha=0.6,
                           connectionstyle='arc3,rad=0.1',
                           ax=ax)
    
    # Draw node labels (truncate long labels)
    node_labels = {}
    for node in G.nodes():
        label = G.nodes[node].get('label', node)
        node_labels[node] = truncate_label(label, max_length=30)
    
    nx.draw_networkx_labels(G, pos, 
                            labels=node_labels,
                            font_size=font_size,
                            font_weight='bold',
                            ax=ax)
    
    # Draw edge labels (relations) - only show for important edges or sample
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', '')
        # Truncate long relations
        relation = truncate_label(relation, max_length=25)
        edge_labels[(u, v)] = relation
    
    # Only draw edge labels if not too many edges
    if len(edge_labels) <= 50:
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels=edge_labels,
                                     font_size=6,
                                     bbox=dict(boxstyle='round,pad=0.3', 
                                              facecolor='white', 
                                              alpha=0.7),
                                     ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    return pos

def visualize_comparison(original_subgraph, refined_subgraph, output_path=None):
    """Visualize both original and refined subgraphs side by side"""
    
    # Convert to NetworkX graphs
    G_original = subgraph_to_networkx(original_subgraph)
    G_refined = subgraph_to_networkx(refined_subgraph)
    
    print(f"Original subgraph: {len(G_original.nodes())} nodes, {len(G_original.edges())} edges")
    print(f"Refined subgraph: {len(G_refined.nodes())} nodes, {len(G_refined.edges())} edges")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 16))
    
    # Visualize original subgraph
    pos_original = visualize_subgraph(G_original, 
                                     "Original Subgraph (Before Refinement)",
                                     figsize=None,
                                     node_size=1500,
                                     font_size=7,
                                     ax=ax1)
    
    # Visualize refined subgraph
    pos_refined = visualize_subgraph(G_refined,
                                    "Refined Subgraph (After Refinement)",
                                    figsize=None,
                                    node_size=1500,
                                    font_size=7,
                                    ax=ax2)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    return G_original, G_refined

def visualize_interaction_history(interaction_history, output_path=None):
    """Visualize each step in the interaction history"""
    num_steps = len(interaction_history)
    
    # Create a grid layout for all steps
    cols = min(3, num_steps)
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15 * cols, 10 * rows))
    if num_steps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, step in enumerate(interaction_history):
        subgraph = step.get("retrieved_subgraph", [])
        num_hops = step.get("num_hops", 0)
        answerable = step.get("answerable", False)
        
        G = subgraph_to_networkx(subgraph)
        
        pos = visualize_subgraph(G,
                                f"Step {idx + 1} (Hops: {num_hops}, Answerable: {answerable})",
                                figsize=None,
                                node_size=1000,
                                font_size=6,
                                ax=axes[idx])
    
    # Hide unused subplots
    for idx in range(num_steps, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Interaction history visualization saved to {output_path}")

def main():
    # Load refinement result
    mode = "after" # or "after"
    if mode == "before":
        name = "refinement_result_0"
    elif mode == "after":
        name = "refinement_result_1"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    result = load_refinement_result(f"/data/haoyuhuang/data/AtlasTune/data/refine_result/{name}.json")
    
    # Extract subgraphs
    original_subgraph = result.get("original_subgraph", [])
    refined_subgraph = result.get("refined_subgraph", [])
    interaction_history = result.get("interaction_history", [])
    query = result.get("query", "")
    
    print(f"Query: {query}")
    print(f"Number of interaction steps: {len(interaction_history)}")
    print()
    
    # Visualize comparison
    print("Visualizing original vs refined subgraph...")
    visualize_comparison(original_subgraph, 
                        refined_subgraph,
                        output_path=f"/data/haoyuhuang/data/AtlasTune/data/refine_result/refinement_comparison_{name}.png")
    
    # Visualize interaction history
    if interaction_history:
        print("\nVisualizing interaction history...")
        visualize_interaction_history(interaction_history,
                                     output_path=f"/data/haoyuhuang/data/AtlasTune/data/refine_result/{name}.png")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()

