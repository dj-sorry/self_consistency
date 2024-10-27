import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import List, Dict, Union, Optional
import numpy as np
import os
from pathlib import Path

class ChainOfThoughtExperiment:
    def __init__(self, 
                 model_name: str = "facebook/opt-1.3b",
                 offload_folder: Optional[str] = None):

        print(f"Loading model: {model_name}")
        
        # offload folder if none
        if offload_folder is None:
            offload_folder = Path("model_offload")
            offload_folder.mkdir(exist_ok=True)
            offload_folder = str(offload_folder)
    
        if any(name in model_name for name in ["opt", "bloom", "gpt2"]):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                offload_folder=offload_folder
            )
            self.is_causal = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
                offload_folder=offload_folder
            )
            self.is_causal = False
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> List[Dict[str, str]]:

        from experiments.data.problems import get_few_shot_examples
        return get_few_shot_examples()

    def generate_cot_prompt(self, question: str) -> str:

        if self.is_causal:
            prompt = "You are a mathematical reasoning expert. Solve these math word problems step by step, showing all calculations clearly:\n\n"
        else:
            prompt = "Solve these math word problems step by step:\n\n"
            
        for example in self.few_shot_examples:
            prompt += f"Question: {example['question']}\n{example['solution']}\n\n"
        
        prompt += f"Question: {question}\nLet's solve this step by step:\n1)"
        
        return prompt

    def generate_reasoning_paths(self, 
                               question: str, 
                               num_samples: int = 5, 
                               temperature: float = 0.7, 
                               max_length: int = 512) -> List[str]:
        
        # Generate multiple reasoning paths for a question

        prompt = self.generate_cot_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        paths = []
        for _ in range(num_samples):
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    num_beams=4,
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=1
                )
                
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if self.is_causal:
                    generated_text = decoded[len(prompt):]
                else:
                    generated_text = decoded
                    
                paths.append(generated_text.strip())
                
            except Exception as e:
                print(f"Error in generation: {str(e)}")
                paths.append("")
                
        return paths

    def evaluate_question(self, 
                         question: str, 
                         correct_answer: str,
                         num_samples: int = 1) -> Dict:
        #Evaluate a single question using both standard CoT and self-consistency
        from experiments.utils.evaluation import (
            extract_final_answer,
            majority_vote,
            check_answer_correctness
        )
        
        # vanilla COT
        standard_path = self.generate_reasoning_paths(question, num_samples=1)[0]
        standard_answer = extract_final_answer(standard_path)
        
        # Self-consistency paths
        paths = self.generate_reasoning_paths(question, num_samples=num_samples)
        answers = [extract_final_answer(path) for path in paths]
        sc_answer = majority_vote(answers)
        
        # Check
        standard_correct = check_answer_correctness(standard_answer, correct_answer)
        sc_correct = check_answer_correctness(sc_answer, correct_answer)
        
        return {
            'standard_correct': standard_correct,
            'sc_correct': sc_correct,
            'standard_path': standard_path,
            'sc_paths': paths,
            'standard_answer': standard_answer,
            'sc_answer': sc_answer
        }