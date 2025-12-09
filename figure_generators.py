import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_overall_performance_figure(fig_name,base_dir,df,metrics):
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(16, 8))

    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, df[metric], width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Approach"], rotation=45, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance Comparison Across Classification Approaches")
    ax.legend()
    plt.tight_layout()
    plt.savefig(base_dir + fig_name )

def generate_classwise_performance_figure(metric_name, file_path, save_path=None):
    """
    Generates a grouped bar chart for classwise metrics
    metric_name: "Precision", "Recall", or "F1-score"
    file_path: path to the classwise CSV file
    save_path: optional path to save the figure (PNG/PDF)
    """
    # Load CSV
    df = pd.read_csv(file_path)

    # Rename class columns to "Stage N"
    rename_map = {
        "class_0": "Stage 0",
        "class_1": "Stage 1",
        "class_2": "Stage 2",
        "class_3": "Stage 3",
        "class_4": "Stage 4"
    }
    df = df.rename(columns=rename_map)

    # Extract model names and stage columns
    models = df["approach"]
    stages = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]

    # Convert to numeric (safety)
    df[stages] = df[stages].apply(pd.to_numeric)

    # Plot grouped bar chart
    x = np.arange(len(models))  # positions for approaches
    width = 0.15                 # bar width

    plt.figure(figsize=(16, 7))

    for i, stage in enumerate(stages):
        plt.bar(x + i*width, df[stage], width, label=stage)

    plt.xticks(x + width*2, models, rotation=45, ha='right')
    plt.ylabel(metric_name)
    plt.title(f"Classwise {metric_name} for All Models", fontsize=14)
    plt.legend(title="Cancer Stage")
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


    


if __name__ == "__main__":
    base_dir = ""

    file_paths = {
        "overall_performance": base_dir + "performance_copy.csv",
        "Precision": base_dir + "precision_classwise.csv",
        "Recall": base_dir + "recall_classwise.csv",
        "F1-score": base_dir + "f1_classwise.csv"
    }

    df = pd.read_csv(file_paths["overall_performance"])
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    generate_overall_performance_figure("overall_performance_macro",base_dir, df, metrics)

    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    generate_overall_performance_figure("overall_performance_weighted",base_dir, df, metrics)

    for metric_name in ["Precision", "Recall", "F1-score"]:
        generate_classwise_performance_figure(
            metric_name,
            file_paths[metric_name],
            save_path=base_dir + f"classwise_{metric_name.lower()}_figure.png"
        )
# python /home/FCAM/nahmed/CSE5819_project/overall_figure_gen.py
