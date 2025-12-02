"""
Visualization Utilities for Graph-Augmented RAG Research
Creates publication-quality figures for papers and presentations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from matplotlib.patches import FancyBboxPatch
import json


class RAGVisualizer:
    """
    Comprehensive visualization toolkit for RAG evaluation
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-paper'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        
        # Set default parameters
        sns.set_palette("husl")
        self.colors = {
            'baseline': '#3498db',
            'ga_rag': '#2ecc71',
            'improvement': '#e74c3c',
            'neutral': '#95a5a6'
        }
    
    def plot_metric_comparison(
        self,
        baseline_metrics: List[float],
        ga_rag_metrics: List[float],
        metric_name: str,
        query_labels: List[str] = None,
        save_path: str = None
    ):
        """
        Plot side-by-side comparison of a single metric
        
        Args:
            baseline_metrics: Baseline scores
            ga_rag_metrics: GA-RAG scores
            metric_name: Name of metric
            query_labels: Labels for queries
            save_path: Path to save figure
        """
        if query_labels is None:
            query_labels = [f"Q{i+1}" for i in range(len(baseline_metrics))]
        
        x = np.arange(len(query_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, baseline_metrics, width, 
                      label='Baseline RAG', color=self.colors['baseline'],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, ga_rag_metrics, width,
                      label='Graph-Augmented RAG', color=self.colors['ga_rag'],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Query', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'{metric_name} Score', fontweight='bold', fontsize=12)
        ax.set_title(f'{metric_name} Comparison: Baseline vs Graph-Augmented RAG',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(query_labels, rotation=45, ha='right')
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_aggregate_comparison(
        self,
        baseline_avg: Dict[str, float],
        ga_rag_avg: Dict[str, float],
        save_path: str = None
    ):
        """
        Plot aggregate metrics comparison
        
        Args:
            baseline_avg: Average metrics for baseline
            ga_rag_avg: Average metrics for GA-RAG
            save_path: Path to save figure
        """
        metrics = list(baseline_avg.keys())
        baseline_values = list(baseline_avg.values())
        ga_rag_values = list(ga_rag_avg.values())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars1 = ax.bar(x - width/2, baseline_values, width,
                      label='Baseline RAG', color=self.colors['baseline'],
                      alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, ga_rag_values, width,
                      label='Graph-Augmented RAG', color=self.colors['ga_rag'],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add improvement arrows
        for i, (b_val, g_val) in enumerate(zip(baseline_values, ga_rag_values)):
            if g_val > b_val:
                ax.annotate('', xy=(i, g_val), xytext=(i, b_val),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        ax.set_ylabel('Average Score', fontweight='bold', fontsize=12)
        ax.set_title('Average Metrics Comparison: Baseline vs Graph-Augmented RAG',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_improvement_heatmap(
        self,
        improvements: Dict[str, List[float]],
        query_labels: List[str] = None,
        save_path: str = None
    ):
        """
        Plot heatmap showing improvements across queries and metrics
        
        Args:
            improvements: Dict of metric_name -> list of improvement percentages
            query_labels: Labels for queries
            save_path: Path to save figure
        """
        # Create DataFrame
        df = pd.DataFrame(improvements)
        
        if query_labels:
            df.index = query_labels
        else:
            df.index = [f"Query {i+1}" for i in range(len(df))]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Improvement (%)'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title('Improvement Heatmap: GA-RAG vs Baseline (%)',
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
        ax.set_ylabel('Queries', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_knowledge_graph(
        self,
        graph: nx.DiGraph,
        title: str = "Knowledge Graph",
        max_nodes: int = 50,
        save_path: str = None,
        show: bool = True
    ):
        """
        Visualize knowledge graph
        
        Args:
            graph: NetworkX graph
            title: Plot title
            max_nodes: Maximum nodes to display
            save_path: Path to save figure
        """
        # Limit nodes if too large
        if graph.number_of_nodes() > max_nodes:
            # Get subgraph with highest degree nodes
            top_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = graph.subgraph([node for node, _ in top_nodes])
        else:
            subgraph = graph
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use spring layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Draw nodes with different colors based on degree
        node_colors = [subgraph.degree(node) for node in subgraph.nodes()]
        
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_color=node_colors,
                              node_size=500,
                              cmap='viridis',
                              alpha=0.7,
                              ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos,
                              edge_color='gray',
                              alpha=0.5,
                              arrows=True,
                              arrowsize=15,
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos,
                               font_size=8,
                               font_weight='bold',
                               ax=ax)
        
        # Draw edge labels (relations)
        edge_labels = nx.get_edge_attributes(subgraph, 'relation')
        nx.draw_networkx_edge_labels(subgraph, pos,
                                     edge_labels=edge_labels,
                                     font_size=6,
                                     ax=ax)
        
        ax.set_title(f'{title}\n({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)',
                    fontweight='bold', fontsize=14, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_pipeline_diagram(
        self,
        save_path: str = None
    ):
        """
        Create pipeline architecture diagram
        
        Args:
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Define boxes
        boxes = [
            # Baseline RAG (top)
            {'x': 0.5, 'y': 4.5, 'width': 1.5, 'height': 0.8, 'text': 'Query', 'color': '#ecf0f1'},
            {'x': 2.5, 'y': 4.5, 'width': 1.5, 'height': 0.8, 'text': 'Retrieval', 'color': '#3498db'},
            {'x': 4.5, 'y': 4.5, 'width': 1.5, 'height': 0.8, 'text': 'Generation', 'color': '#3498db'},
            {'x': 6.5, 'y': 4.5, 'width': 1.5, 'height': 0.8, 'text': 'Answer', 'color': '#ecf0f1'},
            
            # GA-RAG (bottom)
            {'x': 0.5, 'y': 2, 'width': 1.5, 'height': 0.8, 'text': 'Query', 'color': '#ecf0f1'},
            {'x': 2.5, 'y': 2, 'width': 1.5, 'height': 0.8, 'text': 'Retrieval', 'color': '#2ecc71'},
            {'x': 4.5, 'y': 2, 'width': 1.5, 'height': 0.8, 'text': 'Graph\nConstruction', 'color': '#2ecc71'},
            {'x': 6.5, 'y': 2, 'width': 1.5, 'height': 0.8, 'text': 'Consistency\nCheck', 'color': '#2ecc71'},
            {'x': 8.5, 'y': 2, 'width': 1.5, 'height': 0.8, 'text': 'Generation', 'color': '#2ecc71'},
        ]
        
        # Draw boxes
        for box in boxes:
            fancy_box = FancyBboxPatch(
                (box['x'], box['y']), box['width'], box['height'],
                boxstyle="round,pad=0.1", 
                facecolor=box['color'],
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(fancy_box)
            
            ax.text(box['x'] + box['width']/2, box['y'] + box['height']/2,
                   box['text'], ha='center', va='center',
                   fontsize=10, fontweight='bold')
        
        # Draw arrows for baseline
        for i in range(3):
            ax.arrow(0.5 + i*2 + 1.5, 4.9, 0.8, 0, 
                    head_width=0.15, head_length=0.1, fc='black', ec='black')
        
        # Draw arrows for GA-RAG
        for i in range(4):
            ax.arrow(0.5 + i*2 + 1.5, 2.4, 0.8, 0,
                    head_width=0.15, head_length=0.1, fc='black', ec='black')
        
        # Add labels
        ax.text(5, 5.8, 'Baseline RAG Pipeline', 
               ha='center', fontsize=14, fontweight='bold', color='#3498db')
        ax.text(5, 3.3, 'Graph-Augmented RAG Pipeline',
               ha='center', fontsize=14, fontweight='bold', color='#2ecc71')
        
        # Add legend box
        legend_text = (
            "Key Difference:\n"
            "• Baseline: Direct retrieval → generation\n"
            "• GA-RAG: Graph reasoning + consistency filtering"
        )
        ax.text(5, 0.5, legend_text,
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def plot_results_summary(
        self,
        comparison_data: Dict[str, Any],
        save_path: str = None
    ):
        """
        Create comprehensive results summary figure
        
        Args:
            comparison_data: Dict with baseline, ga_rag, and improvements
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        baseline_avg = comparison_data['baseline']
        ga_rag_avg = comparison_data['graph_augmented']
        improvements = comparison_data['improvements_pct']
        
        # 1. Average metrics comparison
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = list(baseline_avg.keys())[:4]  # Top 4 metrics
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, [baseline_avg[m] for m in metrics], width,
               label='Baseline', color=self.colors['baseline'], alpha=0.8)
        ax1.bar(x + width/2, [ga_rag_avg[m] for m in metrics], width,
               label='GA-RAG', color=self.colors['ga_rag'], alpha=0.8)
        
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Average Metrics Comparison', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Improvement percentages
        ax2 = fig.add_subplot(gs[0, 2])
        imp_metrics = list(improvements.keys())[:4]
        imp_values = [improvements[m] for m in imp_metrics]
        colors = [self.colors['ga_rag'] if v > 0 else self.colors['improvement'] 
                 for v in imp_values]
        
        ax2.barh(imp_metrics, imp_values, color=colors, alpha=0.8)
        ax2.set_xlabel('Improvement (%)', fontweight='bold')
        ax2.set_title('Improvement Over Baseline', fontweight='bold', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Summary statistics table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        
        table_data = [
            ['Metric', 'Baseline', 'GA-RAG', 'Improvement (%)'],
            *[[m, f"{baseline_avg[m]:.3f}", f"{ga_rag_avg[m]:.3f}", 
               f"{improvements[m]:+.1f}%"] 
              for m in list(baseline_avg.keys())[:6]]
        ]
        
        table = ax3.table(cellText=table_data, cellLoc='center',
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color improvement column
        for i in range(1, len(table_data)):
            cell = table[(i, 3)]
            value = float(table_data[i][3].rstrip('%'))
            cell.set_facecolor('#2ecc71' if value > 0 else '#e74c3c')
            cell.set_alpha(0.3)
        
        fig.suptitle('Graph-Augmented RAG: Complete Results Summary',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create visualizer
    viz = RAGVisualizer()
    
    # Sample data
    baseline_metrics = [0.65, 0.72, 0.68, 0.70, 0.66]
    ga_rag_metrics = [0.78, 0.85, 0.82, 0.88, 0.79]
    query_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    
    print("Creating sample visualizations...\n")
    
    # 1. Metric comparison
    viz.plot_metric_comparison(
        baseline_metrics,
        ga_rag_metrics,
        "Factual Accuracy",
        query_labels,
        save_path="metric_comparison.png"
    )
    
    # 2. Aggregate comparison
    baseline_avg = {
        "Factual Accuracy": 0.68,
        "Logical Consistency": 0.65,
        "Hallucination Rate": 0.25,
        "Response Coherence": 0.82
    }
    
    ga_rag_avg = {
        "Factual Accuracy": 0.82,
        "Logical Consistency": 0.89,
        "Hallucination Rate": 0.12,
        "Response Coherence": 0.87
    }
    
    viz.plot_aggregate_comparison(
        baseline_avg,
        ga_rag_avg,
        save_path="aggregate_comparison.png"
    )
    
    # 3. Pipeline diagram
    viz.plot_pipeline_diagram(save_path="pipeline_diagram.png")
    
    print("\n✓ All visualizations created successfully!")