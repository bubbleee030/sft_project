import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import logging

def _extract_score(text, metric_name):
    """
    Extracts a score for a given metric from a text string using regex.
    """
    pattern = f"{metric_name}:\s*([\d\.]+)"
    match = re.search(pattern, str(text))
    if match:
        return float(match.group(1))
    return np.nan

def run_evaluation_analysis(prediction_file: str, output_dir: str):
    """
    Analyzes and visualizes scores from a model prediction file.
    """
    logging.info("--- Running Evaluation Result Analysis ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Data ---
    if not os.path.exists(prediction_file):
        logging.error(f"Prediction file not found: {prediction_file}")
        return
    
    try:
        logging.info(f"Loading predictions from: {prediction_file}")
        data = pd.read_json(prediction_file, lines=True)
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return

    if data.empty:
        logging.warning("Prediction file is empty. No analysis to perform.")
        return

    # --- 2. Feature Extraction ---
    logging.info("Extracting scores for 'MatchPoint' and 'TextFormat'...")
    data['match_point_score'] = data['model_resp'].apply(lambda text: _extract_score(text, 'MatchPoint'))
    data['text_format_score'] = data['model_resp'].apply(lambda text: _extract_score(text, 'TextFormat'))

    # --- 3. Print Statistics ---
    match_point_counts = data['match_point_score'].value_counts().sort_index(ascending=False)
    text_format_counts = data['text_format_score'].value_counts().sort_index(ascending=False)

    stats_report = f"""
--- MatchPoint Score Counts ---
{match_point_counts.to_string()}

--- TextFormat Score Counts ---
{text_format_counts.to_string()}

--- Overall Score Statistics ---
{data[['match_point_score', 'text_format_score']].describe().to_string()}
"""
    print(stats_report)
    logging.info(stats_report)

    # --- 4. Visualization ---
    logging.info("Generating score distribution plots...")
    sns.set_theme(style="whitegrid", rc={"font.size":12, "axes.titlesize":16, "axes.labelsize":14})
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Model Score Distribution', fontsize=20)

    # Plot MatchPoint
    sns.histplot(data['match_point_score'].dropna(), kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('MatchPoint Score Distribution')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Count')
    mean_mp = data['match_point_score'].mean()
    axes[0].axvline(mean_mp, color='r', linestyle='--', label=f'Mean: {mean_mp:.2f}')
    axes[0].legend()

    # Plot TextFormat
    sns.histplot(data['text_format_score'].dropna(), kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('TextFormat Score Distribution')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('')
    mean_tf = data['text_format_score'].mean()
    axes[1].axvline(mean_tf, color='r', linestyle='--', label=f'Mean: {mean_tf:.2f}')
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, 'evaluation_scores_distribution.png')
    try:
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Score distribution plot saved to: {save_path}")
        print(f"\nPlot saved to '{save_path}'")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
