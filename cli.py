import argparse
import logging
import os
from datetime import datetime

# Import the run functions from our refactored modules
from src.sft_project.training_pipeline import step1_extract, step2_balance, step3_format
from src.sft_project.inference_pipeline import prepare as prepare_inference
from src.sft_project.analysis import data_visualizer, length_analyzer, eval_analyzer

def setup_logging():
    """Sets up a basic console logger."""
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_directory = 'logs'
    os.makedirs(log_directory, exist_ok=True)
    log_filename = f"cli_run_{log_time}.log"
    log_filepath = os.path.join(log_directory, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler() # Also print to console
        ]
    )

def main():
    """Main function to parse arguments and dispatch commands."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="A unified command-line tool for the SFT data project."
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- 1. Training Data Processing Command ---
    p_train = subparsers.add_parser('process-training-data', help='Run the full pipeline for processing training data.')
    p_train.add_argument('--step1_input_dir', default='./eval_results', help='Input directory for step 1 (extraction).')
    p_train.add_argument('--step1_output_dir', default='./final_classified_output', help='Output directory for step 1, and input for step 2.')
    p_train.add_argument('--step2_output_dir', default='./final_classified_output_new', help='Output directory for step 2, and input for step 3.')
    p_train.add_argument('--step3_output_dir', default='./judge_training_data', help='Output directory for step 3 (final training file).')
    p_train.add_argument('--step3_output_filename', default='judge_finetuning_dataset.jsonl', help='Filename for the final training file.')
    p_train.add_argument('--simplified_chars_file', default='simplified_chars.txt', help='Path to simplified characters file for step 1.')
    p_train.add_argument('--dedup_threshold', type=int, default=2, help='SimHash distance threshold for step 1.')
    p_train.add_argument('--skip_step1', action='store_true', help='Skip step 1 (extraction, deduplication, classification).')
    p_train.add_argument('--skip_step2', action='store_true', help='Skip step 2 (balancing).')
    p_train.add_argument('--skip_step3', action='store_true', help='Skip step 3 (formatting).')
    p_train.add_argument('--no-simhash', action='store_true', help='Ablation: skip SimHash deduplication')
    p_train.add_argument('--keep-simplified', action='store_true', help='Ablation: include simplified Chinese data in training')
    p_train.add_argument('--keep-think', action='store_true', help='Ablation: keep entries containing <think> tags')

    # --- 2. Inference Data Preparation Command ---
    p_infer = subparsers.add_parser('prepare-inference-data', help='Prepare a dataset for model inference.')
    p_infer.add_argument('--input_dir', required=True, help='Source directory containing raw .jsonl files.')
    p_infer.add_argument('--output_dir', default='./judge_training_data', help='Directory to save the output file.')
    p_infer.add_argument('--output_filename', default='judge_inference_dataset.jsonl', help='Filename for the inference dataset.')
    p_infer.add_argument('--limit_per_file', type=int, default=100, help='Maximum number of items to process per input file.')

    # --- 3. Length Analysis Command ---
    p_len = subparsers.add_parser('analyze-length', help='Analyze token lengths of a dataset.')
    p_len.add_argument('--dataset_file', required=True, help='Path to the .jsonl dataset file to analyze.')
    p_len.add_argument('--model_name', required=True, help='Name of the Hugging Face model for tokenization (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct).')
    p_len.add_argument('--hf_token', type=str, default=None, help='Optional Hugging Face token for private models.')

    # --- 4. Visualization Command ---
    p_viz = subparsers.add_parser('visualize-data', help='Generate visualizations from metadata.')
    p_viz.add_argument('--metadata_file', default='./final_classified_output/full_metadata_log.json', help='Path to the full_metadata_log.json file.')
    p_viz.add_argument('--output_dir', default='./visualization_report', help='Directory to save visualization charts.')

    # --- 5. Evaluation Analysis Command ---
    p_eval = subparsers.add_parser('analyze-eval-results', help='Analyze scores from a model prediction file.')
    p_eval.add_argument('--prediction_file', required=True, help='Path to the model_predictions.jsonl file.')
    p_eval.add_argument('--output_dir', default='.', help='Directory to save the analysis plot.')

    args = parser.parse_args()

    # --- Dispatch Command ---
    if args.command == 'process-training-data':
        if not args.skip_step1:
            step1_extract.run_extraction_pipeline(
                input_dir=args.step1_input_dir, 
                output_dir=args.step1_output_dir, 
                simplified_chars_file=args.simplified_chars_file, 
                dedup_threshold=args.dedup_threshold,
                no_simhash=args.no_simhash,
                keep_simplified=args.keep_simplified,
                keep_think=args.keep_think
            )
        if not args.skip_step2:
            step2_balance.run_balancing_pipeline(
                input_dir=args.step1_output_dir, 
                output_dir=args.step2_output_dir,
                keep_simplified=args.keep_simplified
            )
        if not args.skip_step3:
            step3_format.run_formatting_pipeline(
                source_dir=args.step2_output_dir, 
                output_dir=args.step3_output_dir, 
                output_filename=args.step3_output_filename,
                keep_simplified=args.keep_simplified
            )

    elif args.command == 'prepare-inference-data':
        prepare_inference.run_inference_formatting(
            source_dir=args.input_dir, 
            output_dir=args.output_dir, 
            output_filename=args.output_filename, 
            limit_per_file=args.limit_per_file
        )

    elif args.command == 'analyze-length':
        length_analyzer.run_length_analysis(
            dataset_path=args.dataset_file, 
            model_name=args.model_name, 
            hf_token=args.hf_token
        )

    elif args.command == 'visualize-data':
        data_visualizer.run_visualization(
            metadata_file=args.metadata_file, 
            output_dir=args.output_dir
        )

    elif args.command == 'analyze-eval-results':
        eval_analyzer.run_evaluation_analysis(
            prediction_file=args.prediction_file, 
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
