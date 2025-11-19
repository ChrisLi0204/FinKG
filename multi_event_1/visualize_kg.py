"""
Knowledge Graph Visualization Tool
===================================

Creates network visualizations from the multi-event causal knowledge graph.
Generates interactive and static network graphs showing:
- Event → Mechanism → Asset causal chains
- Relationship polarities (positive/negative/neutral)
- Edge weights based on evidence counts
- Node-level statistics and clustering

"""

import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
from collections import defaultdict
from collections import defaultdict


class KGVisualizer:
    """Visualize multi-event causal knowledge graphs."""
    
    def __init__(self, kg_json_path, csv_path=None):
        """
        Initialize visualizer with KG data.
        
        Args:
            kg_json_path: Path to multi_event_causal_kg.json
            csv_path: Path to multi_event_causal_relationships.csv (optional)
        """
        self.kg_json_path = kg_json_path
        self.csv_path = csv_path
        self.kg = None
        self.df = None
        self.G = None
        self.output_dir = 'output/visualizations'
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self._load_kg_json()
        if csv_path and os.path.exists(csv_path):
            self._load_csv()
    
    def _load_kg_json(self):
        """Load knowledge graph from JSON."""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                self.kg = json.load(f)
            print(f"✓ Loaded KG JSON: {self.kg_json_path}")
            print(f"  - Nodes: {self.kg['metadata']['total_nodes']}")
            print(f"  - Edges: {self.kg['metadata']['total_edges']}")
        except Exception as e:
            print(f"❌ Error loading KG JSON: {e}")
    
    def _load_csv(self):
        """Load relationships from CSV."""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded relationships CSV: {self.csv_path}")
            print(f"  - Rows: {len(self.df)}")
            print(f"  - Columns: {list(self.df.columns)}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
    
    def _build_networkx_graph(self, layer_filter=None):
        """
        Build NetworkX graph from KG data.
        
        Args:
            layer_filter: None (all layers) or specific layer (1, 2, 3, etc.)
        """
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.kg['nodes']:
            if layer_filter and node.get('layer') != layer_filter:
                continue
            
            node_id = node['id']
            node_type = node['type']
            node_name = node['name']
            
            G.add_node(
                node_id,
                label=node_name,
                node_type=node_type,
                layer=node.get('layer', 0),
                mentions=node.get('attributes', {}).get('mention_count', 0),
                polarity=node.get('attributes', {}).get('dominant_polarity', 'neutral')
            )
        
        # Add edges
        for edge in self.kg['edges']:
            source = edge['source']
            target = edge['target']
            relation = edge['relation']
            polarity = edge['polarity']
            evidence_count = edge['evidence_count']
            
            # Only add if both nodes exist in filtered graph
            if source in G.nodes() and target in G.nodes():
                G.add_edge(
                    source,
                    target,
                    relation=relation,
                    polarity=polarity,
                    weight=min(evidence_count / 10 + 0.5, 5),  # Normalize weight
                    evidence_count=evidence_count
                )
        
        self.G = G
        return G
    
    def _get_node_color(self, node_id):
        """Get node color based on type/layer."""
        if not self.G:
            return 'lightblue'
        
        layer = self.G.nodes[node_id].get('layer', 0)
        
        # Color by layer
        if layer == 1:  # Events
            return '#FF6B6B'  # Red
        elif layer == 2:  # Mechanisms
            return '#4ECDC4'  # Teal
        elif layer == 3:  # Assets
            return '#95E1D3'  # Light green
        elif layer == 4:  # Outcomes
            return '#FFD93D'  # Yellow
        else:
            return '#D0D0D0'  # Gray - Other
    
    def _get_node_shape(self, node_id):
        """Get node shape based on layer for visual differentiation."""
        if not self.G:
            return 'o'
        
        layer = self.G.nodes[node_id].get('layer', 0)
        
        # Different shapes per layer
        if layer == 1:  # Events - diamond
            return 'D'
        elif layer == 2:  # Mechanisms - square
            return 's'
        elif layer == 3:  # Assets - circle
            return 'o'
        elif layer == 4:  # Outcomes - triangle
            return '^'
        else:
            return 'o'  # Default circle
    
    def _get_edge_color(self, source, target):
        """Get edge color based on polarity."""
        edge_data = self.G.edges[source, target]
        polarity = edge_data.get('polarity', 'neutral')
        
        if polarity == 'positive':
            return '#2ECC71'  # Green
        elif polarity == 'negative':
            return '#E74C3C'  # Red
        else:
            return '#95A5A6'  # Gray
    
    def _get_hierarchical_layout(self, G):
        """Create hierarchical layout with layers separated vertically."""
        pos = {}
        
        # Group nodes by layer
        layers = {}
        for node in G.nodes():
            layer = G.nodes[node].get('layer', 0)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        # Position each layer
        layer_y_positions = {
            1: 1.0,      # Events at top
            2: 0.5,      # Mechanisms in middle
            3: 0.0,      # Assets at bottom
            4: -0.5      # Outcomes below
        }
        
        for layer, nodes in sorted(layers.items()):
            y_pos = layer_y_positions.get(layer, 0)
            num_nodes = len(nodes)
            
            if num_nodes == 1:
                # Single node - center it
                pos[nodes[0]] = (0, y_pos)
            else:
                # Multiple nodes - spread horizontally within layer
                x_spacing = 2.5 / max(1, num_nodes - 1)
                for i, node in enumerate(nodes):
                    x_pos = (i - num_nodes/2 + 0.5) * x_spacing
                    pos[node] = (x_pos, y_pos)
        
        # Apply spring layout within each layer to avoid overlaps
        # but preserve layer separation
        try:
            # Fine-tune positions with spring layout that respects boundaries
            pos = nx.spring_layout(
                G, 
                pos=pos,
                fixed=[],  # Don't fix any nodes, allow adjustment
                k=1.5,     # Smaller k for tighter layout
                iterations=100,
                seed=42
            )
        except:
            # Fallback if spring_layout with pos fails
            pass
        
        return pos
    
    def visualize_full_graph(self, figsize=(28, 18)):
        """Visualize entire knowledge graph."""
        if not self.kg:
            print("❌ No KG loaded")
            return
        
        self._build_networkx_graph()
        
        print(f"\nVisualizing full graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Use hierarchical layout
        pos = self._get_hierarchical_layout(self.G)
        
        # Add layer background regions
        ax.axhspan(0.85, 1.15, alpha=0.1, color='red', label='Layer 1')
        ax.axhspan(0.35, 0.65, alpha=0.1, color='teal', label='Layer 2')
        ax.axhspan(-0.15, 0.15, alpha=0.1, color='green', label='Layer 3')
        
        # Node colors and sizes
        node_colors = [self._get_node_color(node) for node in self.G.nodes()]
        node_sizes = []
        for node in self.G.nodes():
            mentions = self.G.nodes[node].get('mentions', 1)
            # Larger sizes: baseline 600, scale by mentions
            size = min(mentions * 15 + 600, 8000)
            node_sizes.append(size)
        
        # Draw nodes by layer with different shapes
        layers = set(self.G.nodes[node].get('layer', 0) for node in self.G.nodes())
        
        for layer in sorted(layers):
            layer_nodes = [node for node in self.G.nodes() if self.G.nodes[node].get('layer', 0) == layer]
            layer_colors = [self._get_node_color(node) for node in layer_nodes]
            layer_sizes = [node_sizes[list(self.G.nodes()).index(node)] for node in layer_nodes]
            layer_shape = self._get_node_shape(layer_nodes[0]) if layer_nodes else 'o'
            
            # Map matplotlib markers for each shape
            marker_map = {'D': 'd', 's': 's', 'o': 'o', '^': '^'}
            marker = marker_map.get(layer_shape, 'o')
            
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=layer_nodes,
                node_color=layer_colors,
                node_size=layer_sizes,
                node_shape=marker,
                alpha=0.85,
                edgecolors='black',
                linewidths=2,
                ax=ax
            )
        
        # Draw edges by polarity
        for edge in self.G.edges():
            source, target = edge
            color = self._get_edge_color(source, target)
            edge_data = self.G.edges[source, target]
            polarity = edge_data.get('polarity', 'neutral')
            style = 'solid' if polarity == 'positive' else ('dashed' if polarity == 'negative' else 'dotted')
            weight = edge_data.get('weight', 1)
            
            nx.draw_networkx_edges(
                self.G, pos,
                edgelist=[edge],
                edge_color=color,
                width=weight,
                style=style,
                alpha=0.6,
                connectionstyle="arc3,rad=0.1",
                ax=ax,
                arrows=True,
                arrowsize=15,
                arrowstyle='->'
            )
        
        # Labels - with better positioning to minimize overlap
        labels = {node: self.G.nodes[node]['label'] for node in self.G.nodes()}
        
        # Use adjustText-like approach: draw labels with offsets
        from matplotlib.patheffects import withStroke
        
        for node, (x, y) in pos.items():
            label = labels[node]
            ax.text(
                x, y, label,
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                ),
                path_effects=[withStroke(linewidth=2, foreground='white')]
            )
        
        ax.set_title("Multi-Event Causal Knowledge Graph", fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Legend - enhanced with shapes
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF6B6B', markersize=12, 
                   label='Layer 1: Events', markeredgecolor='black', markeredgewidth=2),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#4ECDC4', markersize=12, 
                   label='Layer 2: Mechanisms', markeredgecolor='black', markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#95E1D3', markersize=12, 
                   label='Layer 3: Assets', markeredgecolor='black', markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=0),  # Spacer
            mpatches.Patch(facecolor='white', edgecolor='#2ECC71', linewidth=2.5, label='Positive Impact'),
            mpatches.Patch(facecolor='white', edgecolor='#E74C3C', linewidth=2.5, label='Negative Impact'),
            mpatches.Patch(facecolor='white', edgecolor='#95A5A6', linewidth=2.5, linestyle=':', label='Neutral Impact'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '01_full_graph.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def visualize_by_layer(self):
        """Visualize each layer separately."""
        if not self.kg:
            print("❌ No KG loaded")
            return
        
        # Determine unique layers
        layers = set(node.get('layer', 0) for node in self.kg['nodes'])
        
        for layer in sorted(layers):
            self._build_networkx_graph(layer_filter=layer)
            
            if self.G.number_of_nodes() == 0:
                continue
            
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Use better layout based on graph size
            num_nodes = self.G.number_of_nodes()
            if num_nodes <= 5:
                # Circular layout for very small graphs
                pos = nx.circular_layout(self.G)
            elif num_nodes <= 15:
                # Spring layout with more spacing for small graphs
                pos = nx.spring_layout(self.G, k=3.0, iterations=150, seed=42)
            else:
                # Spring layout with good spacing
                pos = nx.spring_layout(self.G, k=2.5, iterations=100, seed=42)
            
            # Node styling - ensure nodes are always visible
            node_colors = [self._get_node_color(node) for node in self.G.nodes()]
            # Larger, more consistent node sizes
            node_sizes = [max(self.G.nodes[node].get('mentions', 1) * 80 + 800, 1200) for node in self.G.nodes()]
            
            # Draw
            nx.draw_networkx_nodes(
                self.G, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.85,
                edgecolors='black',
                linewidths=2.5,
                ax=ax
            )
            
            # Edges
            for edge in self.G.edges():
                source, target = edge
                color = self._get_edge_color(source, target)
                edge_data = self.G.edges[source, target]
                style = 'solid' if edge_data.get('polarity') == 'positive' else ('dashed' if edge_data.get('polarity') == 'negative' else 'dotted')
                
                nx.draw_networkx_edges(
                    self.G, pos,
                    edgelist=[edge],
                    edge_color=color,
                    width=2.5,
                    style=style,
                    alpha=0.7,
                    ax=ax,
                    arrows=True,
                    arrowsize=20,
                    connectionstyle="arc3,rad=0.1"
                )
            
            # Labels - ensure readable
            labels = {node: self.G.nodes[node]['label'] for node in self.G.nodes()}
            from matplotlib.patheffects import withStroke
            
            for node, (x, y) in pos.items():
                label = labels[node]
                ax.text(
                    x, y, label,
                    fontsize=10,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.35',
                        facecolor='white',
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.85
                    ),
                    path_effects=[withStroke(linewidth=2, foreground='white')]
                )
            
            layer_names = {1: "Events", 2: "Mechanisms", 3: "Assets", 4: "Outcomes"}
            layer_name = layer_names.get(layer, f"Layer {layer}")
            
            ax.set_title(f"Layer {layer}: {layer_name}", fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, f'02_layer_{layer}_{layer_name.lower()}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            plt.close()
    
    def visualize_causal_chains(self):
        """Visualize top causal chains: Event → Mechanism → Asset."""
        if not self.df is None:
            # Count chain occurrences
            chains = defaultdict(int)
            for _, row in self.df.iterrows():
                if pd.notna(row['source']) and pd.notna(row['target']):
                    chain_key = (str(row['source']), str(row['target']), row.get('polarity', 'neutral'))
                    chains[chain_key] += 1
            
            # Get top chains
            top_chains = sorted(chains.items(), key=lambda x: x[1], reverse=True)[:15]
            
            # Visualize
            fig, ax = plt.subplots(figsize=(14, 10))
            
            chains_list = [f"{c[0][0][:20]} → {c[0][1][:20]}" for c in top_chains]
            counts = [c[1] for c in top_chains]
            colors = ['#2ECC71' if c[0][2] == 'positive' else '#E74C3C' if c[0][2] == 'negative' else '#95A5A6' for c in top_chains]
            
            y_pos = range(len(chains_list))
            ax.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(chains_list, fontsize=9)
            ax.set_xlabel('Evidence Count', fontsize=12, fontweight='bold')
            ax.set_title('Top 15 Causal Chains by Evidence Count', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, '03_top_causal_chains.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            plt.close()
    
    def visualize_polarity_distribution(self):
        """Visualize polarity distribution across relationships."""
        if not self.df is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Overall polarity
            polarity_counts = self.df['polarity'].value_counts()
            colors_pie = ['#2ECC71', '#E74C3C', '#95A5A6']
            axes[0].pie(
                polarity_counts.values,
                labels=polarity_counts.index,
                autopct='%1.1f%%',
                colors=colors_pie[:len(polarity_counts)],
                startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            axes[0].set_title('Overall Relationship Polarity', fontsize=12, fontweight='bold')
            
            # By source type
            polarity_by_source = pd.crosstab(self.df['source_type'], self.df['polarity'])
            polarity_by_source.plot(
                kind='bar',
                ax=axes[1],
                color=['#2ECC71', '#E74C3C', '#95A5A6'],
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5
            )
            axes[1].set_title('Polarity by Source Type', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Source Type', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
            axes[1].legend(title='Polarity', fontsize=10)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, '04_polarity_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            plt.close()
    
    def visualize_node_importance(self):
        """Visualize node importance based on degree and evidence."""
        if not self.G:
            self._build_networkx_graph()
        
        # Calculate centrality metrics - FIXED: use undirected measure
        degree_centrality = nx.degree_centrality(self.G.to_undirected())
        
        # Get evidence counts from edges
        edge_weights = defaultdict(float)
        for source, target, data in self.G.edges(data=True):
            weight = data.get('weight', 1)
            edge_weights[source] += weight
            edge_weights[target] += weight
        
        # Combine centrality and edge weights
        importance = {}
        for node in self.G.nodes():
            centrality_score = degree_centrality.get(node, 0) * 100
            edge_score = edge_weights.get(node, 0)
            importance[node] = centrality_score + edge_score
        
        # Get top nodes
        top_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        node_names = [self.G.nodes[node]['label'][:30] for node, _ in top_nodes]
        scores = [score for _, score in top_nodes]
        colors = [self._get_node_color(node) for node, _ in top_nodes]
        
        y_pos = range(len(node_names))
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(node_names, fontsize=10)
        ax.set_xlabel('Importance Score (Centrality + Edge Weight)', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Most Important Nodes', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '05_node_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def visualize_event_impact_summary(self):
        """Create summary visualization of event impacts."""
        if not self.kg:
            print("❌ No KG loaded")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Metadata
        ax_meta = fig.add_subplot(gs[0, :])
        ax_meta.axis('off')
        
        metadata = self.kg['metadata']
        summary_text = f"""
        KNOWLEDGE GRAPH SUMMARY
        
        Total Nodes: {metadata['total_nodes']} | Total Edges: {metadata['total_edges']}
        Event Types: {', '.join(metadata['event_types'])}
        Architecture: {metadata['architecture']}
        Created: {metadata['created_at'][:10]}
        Description: {metadata['description']}
        """
        
        ax_meta.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Node type distribution
        ax1 = fig.add_subplot(gs[1, 0])
        node_types = defaultdict(int)
        for node in self.kg['nodes']:
            node_types[node['type']] += 1
        
        ax1.bar(list(node_types.keys()), list(node_types.values()), color='skyblue', edgecolor='black', linewidth=1.5)
        ax1.set_title('Node Type Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Relation distribution
        ax2 = fig.add_subplot(gs[1, 1])
        relations = defaultdict(int)
        for edge in self.kg['edges']:
            relations[edge['relation']] += 1
        
        ax2.bar(list(relations.keys()), list(relations.values()), color='lightcoral', edgecolor='black', linewidth=1.5)
        ax2.set_title('Edge Relation Distribution', fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Polarity distribution
        ax3 = fig.add_subplot(gs[2, 0])
        polarities = defaultdict(int)
        for edge in self.kg['edges']:
            polarities[edge['polarity']] += 1
        
        colors = {'positive': '#2ECC71', 'negative': '#E74C3C', 'neutral': '#95A5A6'}
        ax3.pie(list(polarities.values()), labels=list(polarities.keys()),
               colors=[colors.get(k, 'gray') for k in polarities.keys()],
               autopct='%1.1f%%', startangle=90)
        ax3.set_title('Polarity Distribution', fontweight='bold')
        
        # Evidence distribution
        ax4 = fig.add_subplot(gs[2, 1])
        evidence_counts = [edge['evidence_count'] for edge in self.kg['edges']]
        ax4.hist(evidence_counts, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax4.set_title('Evidence Count Distribution', fontweight='bold')
        ax4.set_xlabel('Evidence Count')
        ax4.set_ylabel('Frequency')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Multi-Event Knowledge Graph Overview', fontsize=16, fontweight='bold', y=0.995)
        
        output_path = os.path.join(self.output_dir, '06_kg_summary.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization types."""
        print("\n" + "="*70)
        print("GENERATING KNOWLEDGE GRAPH VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.visualize_full_graph()
        print()
        
        self.visualize_by_layer()
        print()
        
        self.visualize_causal_chains()
        print()
        
        self.visualize_polarity_distribution()
        print()
        
        self.visualize_node_importance()
        print()
        
        self.visualize_event_impact_summary()
        
        print("\n" + "="*70)
        print(f"ALL VISUALIZATIONS COMPLETE")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Paths
    kg_json = 'output/multi_event_causal_kg.json'
    csv_file = 'output/multi_event_causal_relationships.csv'
    
    # Check if files exist
    if not os.path.exists(kg_json):
        print(f"❌ Error: {kg_json} not found")
        print("Please run multi_event_kg_1.py first to generate the knowledge graph.")
    else:
        # Create visualizer and generate all visualizations
        viz = KGVisualizer(kg_json, csv_file)
        viz.generate_all_visualizations()
