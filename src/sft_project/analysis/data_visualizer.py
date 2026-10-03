import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- Internal Helper Functions ---

def _load_full_metadata(file_path):
    """Loads all data from a single metadata log file."""
    if not os.path.exists(file_path):
        logging.error(f"Error: Metadata log file not found at '{file_path}'.")
        return pd.DataFrame()
    logging.info(f"Loading full metadata from '{file_path}'...")
    df = pd.read_json(file_path)
    logging.info(f"Load complete, found {len(df)} records.")
    return df

def _rename_model(df):
    if df.empty: return df
    model_mapping = {'gen_DeepSeek-R1': 'R1', 'gen_QwQ-32B': 'TAT'}
    logging.info(f"Renaming models: {model_mapping}")
    df['genmodel'] = df['genmodel'].replace(model_mapping)
    return df

def _visualize_status_distribution(df, output_dir):
    if df.empty: return
    logging.info("Visualizing overall data status distribution...")
    status_counts = df['status'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
    plt.title('Overall Status Distribution', fontsize=18, pad=20)
    plt.ylabel('')
    save_path = os.path.join(output_dir, '1_overall_status_distribution.png')
    plt.savefig(save_path); plt.close()
    logging.info(f"Status distribution chart saved to: {save_path}")

def _visualize_error_reasons(df, output_dir):
    if df.empty: return
    logging.info("Visualizing format error reasons...")
    error_df = df[df['status'] == 'FormatError']
    if error_df.empty:
        logging.info("No format errors found, skipping chart.")
        return
    reason_counts = error_df['error_reason'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=reason_counts.values, y=reason_counts.index, palette='viridis')
    plt.title('Format Error Reasons Analysis', fontsize=18, pad=20)
    plt.xlabel('Count', fontsize=14); plt.ylabel('Error Reason', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, '2_format_error_reasons.png')
    plt.savefig(save_path); plt.close()
    logging.info(f"Error reasons chart saved to: {save_path}")

def _visualize_rates_table(df, output_dir):
    if df.empty: return
    logging.info("Generating model health analysis table...")
    df['combination'] = df['genmodel'] + " / " + df['judgemodel']
    status_pivot = pd.pivot_table(df, values='internal_id', index='combination', columns='status', aggfunc='count', fill_value=0)
    if 'Valid' not in status_pivot.columns: status_pivot['Valid'] = 0
    if 'FormatError' not in status_pivot.columns: status_pivot['FormatError'] = 0
    if 'Duplicate' not in status_pivot.columns: status_pivot['Duplicate'] = 0
    status_pivot['Total'] = status_pivot.sum(axis=1)
    status_pivot['Error_Rate'] = (status_pivot['FormatError'] / status_pivot['Total'])
    status_pivot['Dup_Rate'] = (status_pivot['Duplicate'] / status_pivot['Total'])
    
    styled_table = (status_pivot[['Valid', 'FormatError', 'Duplicate', 'Total', 'Error_Rate', 'Dup_Rate']] \
                    .sort_values(by='Total', ascending=False) \
                    .style.format({'Error_Rate': '{:.2%}', 'Dup_Rate': '{:.2%}'}) \
                    .background_gradient(cmap='Reds', subset=['Error_Rate']) \
                    .background_gradient(cmap='Blues', subset=['Dup_Rate']) \
                    .bar(color='#5DADE2', subset=['Valid']) \
                    .set_caption("Model Combination Health Analysis (Error & Duplicate Rates)") \
                    .set_properties(**{'text-align': 'center'}))
    
    save_path = os.path.join(output_dir, '3_model_health_rates_table.html')
    styled_table.to_html(save_path, encode_html="utf-8")
    logging.info(f"Model health table saved to: {save_path}")

def _visualize_model_pairs_heatmap(df, output_dir):
    if df.empty: return
    logging.info("Visualizing model combination heatmap...")
    pivot_table = df.pivot_table(index='genmodel', columns='judgemodel', values='internal_id', aggfunc='count', fill_value=0)
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="viridis", linewidths=.5)
    plt.title('GenModel vs JudgeModel Heatmap (Valid Data Count)', fontsize=18, pad=20)
    plt.xlabel('JudgeModel', fontsize=14); plt.ylabel('GenModel', fontsize=14)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(output_dir, '4_valid_data_heatmap.png')
    plt.savefig(save_path); plt.close()
    logging.info(f"Model combination heatmap saved to: {save_path}")

def _visualize_mission_combinations(df, output_dir):
    if df.empty: return
    logging.info("Analyzing model distribution per mission...")
    missions = df['mission'].unique()
    for mission in missions:
        mission_df = df[df['mission'] == mission]
        if mission_df.empty: continue
        pivot_df = mission_df.pivot_table(index='genmodel', columns='judgemodel', values='internal_id', aggfunc='count', fill_value=0)
        if pivot_df.empty: continue
        pivot_df.plot(kind='bar', stacked=True, figsize=(14, 8), cmap='tab20c', width=0.8)
        plt.title(f'Mission {mission} Model Combination Distribution', fontsize=18, pad=20)
        plt.xlabel('GenModel', fontsize=14); plt.ylabel('Valid Data Count', fontsize=14)
        plt.xticks(rotation=45, ha='right'); plt.legend(title='JudgeModel', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'5_mission_{mission}_distribution.png')
        plt.savefig(save_path); plt.close()
        logging.info(f"Chart for mission '{mission}' saved to: {save_path}")

# --- Main Entry Function ---
def run_visualization(metadata_file: str, output_dir: str):
    """
    Orchestrates the data visualization and analysis process.
    """
    logging.info("--- Running Data Visualization & Analysis Pipeline ---")
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign display

    # It's good practice to check for a font that supports CJK characters
    # This is a simple check, a more robust solution might be needed
    try:
        from matplotlib.font_manager import fontManager
        # Check for a common CJK font, if not found, it will default and might show squares.
        if not any(['Heiti TC' in f.name or 'Microsoft JhengHei' in f.name for f in fontManager.ttflist]):
            logging.warning("No CJK font (like Heiti TC or Microsoft JhengHei) found. Chart labels might not display correctly.")
    except ImportError:
        logging.warning("Could not import fontManager to check for CJK fonts.")

    full_df = _load_full_metadata(metadata_file)
    
    if not full_df.empty:
        full_df = _rename_model(full_df)
        
        # Run analyses on the full dataset
        _visualize_status_distribution(full_df, output_dir)
        _visualize_error_reasons(full_df, output_dir)
        _visualize_rates_table(full_df, output_dir)
        
        # Filter for valid data for deeper analysis
        valid_df = full_df[full_df['status'] == 'Valid'].copy()
        if not valid_df.empty:
            logging.info(f"Filtered to {len(valid_df)} valid items for detailed analysis.")
            _visualize_model_pairs_heatmap(valid_df, output_dir)
            _visualize_mission_combinations(valid_df, output_dir)
        else:
            logging.warning("No valid data available for combination and mission analysis.")

        logging.info("--- All visualization tasks complete ---")
        print(f"\nAnalysis report and charts generated successfully! Please check the '{output_dir}' directory.")
    else:
        logging.warning("No data was loaded, terminating visualization process.")
