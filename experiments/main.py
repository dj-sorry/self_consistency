from typing import List, Dict
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.experiment import ChainOfThoughtExperiment
from data.problems import load_addsum_examples

def run_experiments(
    model_names: List[str], 
    num_samples: int = 5,
    temperatures: List[float] = [0.7],
    offload_folder: str = None
) -> Dict:
    """Run experiments across multiple models and parameters.
    
    Args:
        model_names: List of model names
        num_samples: Number of samples for self-consistency (only tested on 1 at this point")
        temperatures: List of temperature values to test
        offload_folder: Directory to store model weights when using disk offloading
    """
    # Create default offload folder if none provided
    if offload_folder is None:
        offload_folder = Path("model_offload")
        offload_folder.mkdir(exist_ok=True)
        offload_folder = str(offload_folder)
    
    results = {}
    
    for model_name in model_names:
        print(f"\nTesting model: {model_name}")
        try:
            # Initialize experiment with offload folder
            experiment = ChainOfThoughtExperiment(
                model_name=model_name,
                offload_folder=offload_folder
            )
            
            # Results storage
            results[model_name] = {
                temp: {'standard': [], 'self_consistency': []} 
                for temp in temperatures
            }
            
            # Load test
            problems = load_addsum_examples()
            
            # Run experiments for each temperature
            for temp in temperatures:
                print(f"\nRunning experiments with temperature {temp}")
                
                # Initialize progress bar for current temperature
                with tqdm(total=len(problems), desc=f"Processing temp={temp}") as pbar:
                        for problem in problems:
                            print(f"Starting evaluation for question: {problem['question']}")
                            eval_results = experiment.evaluate_question(
                                question=problem["question"],
                                correct_answer=problem["answer"],
                                num_samples=num_samples
                            )
                            print(f"Finished evaluation for question: {problem['question']}")

                                                
                        # Store results
                        results[model_name][temp]['standard'].append(
                            eval_results['standard_correct']
                        )
                        results[model_name][temp]['self_consistency'].append(
                            eval_results['sc_correct']
                        )
                        
                        print(f"\nQuestion: {problem['question']}")
                        print(f"Correct answer: {problem['answer']}")
                        print(f"Model answer: {eval_results['sc_answer']}")
                        print(f"Correct: {eval_results['sc_correct']}")
                        
                        pbar.update(1)
                
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    # distilbert-base-uncased and so on will not work because its not seq2seq
    models = ["t5-small"]
    #models = ["facebook/opt-1.3b"]
    
    results = run_experiments(
        model_names=models,
        num_samples=5,
        temperatures=[0.7],
        offload_folder="model_offload"
    )
    
    print("\n=== Final Results ===")
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        for temp, temp_results in model_results.items():
            std_acc = np.mean(temp_results['standard']) if temp_results['standard'] else 0
            sc_acc = np.mean(temp_results['self_consistency']) if temp_results['self_consistency'] else 0
            
            print(f"Temperature: {temp}")
            print(f"Standard Chain-of-Thought Accuracy: {std_acc:.2%}")
            print(f"Self-Consistency Accuracy: {sc_acc:.2%}")
            print(f"Improvement: {(sc_acc - std_acc):.2%}")
