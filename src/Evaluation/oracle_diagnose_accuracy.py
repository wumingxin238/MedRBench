import os
import json
import logging
import argparse
import multiprocessing
from multiprocessing import Queue, Manager

from metrics.outcome_accuracy_eval import eval_accuracy
from metrics.utils import get_eval_model
from eval_io import load_model_outputs

# Configuration constants
NUM_WORKERS = 4  # Number of parallel worker processes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def extract_answer_content(text):
    """Extract answer content from model output if it contains a specific format."""
    if '### Answer' in text:
        # Extract content after '### Answer', removing newlines and colons
        return text.split('### Answer')[-1].replace('\n', '').replace(':', '')
    return text


def evaluate_case(case_data, output_directory, model_name, evaluation_model):
    """Evaluate a single case and save results."""
    logger.info(f'Evaluating case {case_data["id"]} for model {model_name}')

    try:
        # Get ground truth and model prediction
        ground_truth = case_data['generate_case']['diagnosis_results']
        model_prediction_raw = case_data['results']['content']
        
        # Extract the answer part if it contains the specific format
        model_prediction = extract_answer_content(model_prediction_raw)
        
        # Evaluate accuracy using imported function
        is_accurate = eval_accuracy(
            pred_outcome_answer=model_prediction, 
            gt_outcome_answer=ground_truth,
            evaluation_model=evaluation_model
        )

        # Store evaluation results
        case_data['accuracy'] = is_accurate

        # Save results to file
        output_file = os.path.join(output_directory, f'{case_data["id"]}.json')
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logger.error(f'Error processing case {case_data["id"]}: {str(e)}')


def worker_process(task_queue):
    """Process evaluation tasks from a queue."""
    while not task_queue.empty():
        try:
            case_data, output_directory, model_name, evaluation_model = task_queue.get()
            evaluate_case(case_data, output_directory, model_name, evaluation_model)
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")


def main(
    model_name,
    patient_case_filepath,
    model_output_filepath,
    output_directory,
    use_parallel=True,
    embedded_outputs=False,
    evaluation_model=None,
    num_workers=NUM_WORKERS,
):
    """Orchestrate the evaluation process for a specific model."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load patient cases and model outputs
    with open(patient_case_filepath, 'r', encoding='utf-8') as f:
        patient_cases = json.load(f)
    
    model_outputs = load_model_outputs(
        model_output_filepath, model_name, embedded=embedded_outputs
    )
    eval_model = get_eval_model(evaluation_model)
    logger.info(f'Evaluator model: {eval_model} (backend={os.environ.get("EVAL_BACKEND", "openai")})')
    
    # Identify cases that need to be processed
    cases_to_evaluate = []
    completed_cases = os.listdir(output_directory)
    completed_case_ids = [name.split('.')[0] for name in completed_cases]
    
    for case_id in patient_cases.keys():
        if case_id not in completed_case_ids and case_id in model_outputs and model_name in model_outputs[case_id]:
            case_data = patient_cases[case_id].copy()  # Create a copy to avoid modifying the original
            case_data['id'] = case_id
            case_data['results'] = model_outputs[case_id][model_name]
            cases_to_evaluate.append(case_data)    
    
    logger.info(f'Total cases to evaluate: {len(cases_to_evaluate)}')
    
    if use_parallel and len(cases_to_evaluate) > 0:
        # Parallel processing approach
        manager = Manager()
        task_queue = manager.Queue()
        
        # Add all tasks to queue
        for case_data in cases_to_evaluate:
            task_queue.put((case_data, output_directory, model_name, eval_model))

        # Create and start worker processes
        processes = []
        worker_count = min(num_workers, len(cases_to_evaluate))
        logger.info(f"Starting {worker_count} worker processes")
        
        for _ in range(worker_count):
            process = multiprocessing.Process(target=worker_process, args=(task_queue,))
            process.start()
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.join()
    else:
        # Sequential processing approach
        logger.info("Processing cases sequentially")
        for case_data in cases_to_evaluate:
            evaluate_case(case_data, output_directory, model_name, eval_model)
    
    logger.info(f"Evaluation completed for model {model_name}")


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on diagnose tasks')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['qwq', 'o3-mini', 'gemini2-ft', 'deepseek-r1', 'baichuan-m1', 'qwen3-14b', 'qwen3-14b-thinking', 'qwen3-8b'],
                      help='Model to evaluate')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run sequentially instead of using parallel processing')
    parser.add_argument('--output-dir', type=str, default='./acc_results',
                      help='Base directory for evaluation results')
    parser.add_argument('--patient-cases', type=str,
                      default='../../data/MedRBench/test_cases.json',
                      help='Path to patient cases file (35-sample subset: test_cases.json)')
    parser.add_argument('--model-outputs', type=str,
                      default='../Inference/oracle_diagnosis_gemini.json',
                      help='Path to model outputs file')
    parser.add_argument('--embedded-outputs', action='store_true',
                      help='Model outputs are embedded in the same JSON as inference (e.g. oracle_diagnosis_gemini.json)')
    parser.add_argument('--eval-model', type=str, default=None,
                      help='Evaluator/judge model (overrides EVAL_MODEL env)')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                      help='Parallel worker processes (default 4)')
    
    args = parser.parse_args()
    
    # Define input and output file paths
    model_output_filepath = args.model_outputs
    patient_case_filepath = args.patient_cases
    output_directory = f'{args.output_dir}/{args.model}'
    
    # Run main evaluation process
    main(
        args.model, 
        patient_case_filepath, 
        model_output_filepath, 
        output_directory, 
        not args.sequential,
        embedded_outputs=args.embedded_outputs,
        evaluation_model=args.eval_model,
        num_workers=max(1, args.workers),
    )