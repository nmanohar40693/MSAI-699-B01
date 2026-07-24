import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

RESULTS_DIR = "/Users/naveenmanohar/capstone_experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def render_table_as_png(df, filepath):
    # Adjust width dynamically based on number of columns to prevent text clipping
    width = max(10, len(df.columns) * 1.5)
    fig, ax = plt.subplots(figsize=(width, len(df) * 0.4 + 1.0))
    ax.axis('tight')
    ax.axis('off')
    
    # Render table using matplotlib
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Apply APA 7 styling (horizontal lines only, bold headers)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    table.scale(1.0, 1.5)
    
    # Style cells
    for (row, col), cell in table.get_celld().items():
        # Remove all borders initially
        cell.set_edgecolor('none')
        if row == 0:
            cell.set_text_props(weight='bold')
            # Add line below header
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
            cell.visible_edges = 'B'
        elif row == len(df):
            # Add line below the last row
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
            cell.visible_edges = 'B'
            
    # Add top border on header
    for col in range(len(df.columns)):
        cell = table[0, col]
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
        cell.visible_edges = 'TB'
        
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rendered: {filepath}")

def main():
    # 1. Table 1: Embedding Model Comparison
    t1_data = {
        "Embedding Model": ["all-MiniLM-L6-v2", "microsoft/codebert-base", "sentence-transformers/all-mpnet-base-v2"],
        "Precision@5": ["0.180", "0.000", "0.200"],
        "Recall@5": ["0.850", "0.000", "0.900"],
        "F1-Score": ["0.295", "0.000", "0.324"],
        "MRR": ["0.683", "0.000", "0.750"],
        "Latency (s)": ["0.028", "0.046", "0.126"],
        "Indexing (s)": ["0.224", "0.374", "0.374"],
        "Peak Memory (MB)": ["523.89", "1001.86", "1180.02"]
    }
    df1 = pd.DataFrame(t1_data)
    render_table_as_png(df1, os.path.join(RESULTS_DIR, "table_1_embedding_comparison.png"))

    # 2. Table 2: Optimized Hyperparameters
    t2_data = {
        "Parameter": [
            "Top-K Retrieval", 
            "Similarity Threshold", 
            "Maximum Graph Depth", 
            "Test-Class Relationship Weight", 
            "Issue Resolution Relationship Weight", 
            "Modified File Relationship Weight", 
            "Maximum Context Length (characters)"
        ],
        "Optimized Value": ["5", "0.2330", "3", "2.3019", "2.4364", "1.8027", "2,010"],
        "Description": [
            "Number of initial semantic search entry points",
            "Minimum cosine similarity required for seed nodes",
            "Maximum BFS graph traversal depth",
            "Traversal weight applied to test class relationships",
            "Traversal weight applied to issue-resolution relationships",
            "Traversal weight applied to modified file relationships",
            "Maximum character budget for the constructed context"
        ]
    }
    df2 = pd.DataFrame(t2_data)
    render_table_as_png(df2, os.path.join(RESULTS_DIR, "table_2_optimized_hyperparameters.png"))


    # 3. Table 3: Performance Comparison
    t3_data = {
        "Performance Metric": [
            "Mean Context Length (characters)",
            "Total Input Tokens",
            "Total Output Tokens",
            "Mean Downstream Latency (s)",
            "Code Solution Rate"
        ],
        "Baseline (Default)": ["660.50", "2,372", "1,508", "4.962", "40.0%"],
        "Optimized": ["757.40", "2,730", "780", "27.013", "40.0%"],
        "Absolute Change": ["+96.90", "+358", "-728", "+22.051", "0.00"],
        "Percentage Change": ["+14.67%", "+15.09%", "-48.28%", "+444.40%", "0.00%"]
    }
    df3 = pd.DataFrame(t3_data)
    render_table_as_png(df3, os.path.join(RESULTS_DIR, "table_3_performance_comparison.png"))

    # 4. Table 4: Engineered Features
    t4_data = {
        "Feature": [
            "semantic_similarity",
            "symbol_match",
            "graph_distance",
            "traceability_degree",
            "test_source_link",
            "lifecycle_stage_match",
            "artifact_type_match",
            "version_recency"
        ],
        "Data Source": [
            "Semantic retrieval",
            "Source code analysis",
            "Project knowledge graph",
            "Project knowledge graph",
            "Project knowledge graph",
            "Lifecycle stage information",
            "Repository metadata",
            "Repository metadata"
        ],
        "Purpose": [
            "Measures semantic relevance",
            "Detects artifact references",
            "Measures graph proximity",
            "Captures graph connectivity",
            "Identifies test-source links",
            "Matches lifecycle stage",
            "Matches artifact type",
            "Represents version recency"
        ]
    }
    df4 = pd.DataFrame(t4_data)
    render_table_as_png(df4, os.path.join(RESULTS_DIR, "table_4_engineered_features.png"))

if __name__ == "__main__":
    main()
