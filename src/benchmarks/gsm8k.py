import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from src.benchmarks.base_benchmark import BaseBenchmark
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class GSM8KBenchmark(BaseBenchmark):
    """GSM8K (Grade School Math 8K) benchmark implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.dataset = None
        self.num_shots = config.get("num_shots", 3)
        self.split = config.get("split", "test")
        self.dataset_fraction = config.get("dataset_fraction", 1.0)
        self.generation_kwargs = {
            "max_new_tokens": config.get("max_new_tokens", 256),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 50)
        }
        
    def load_dataset(self) -> None:
        """Load the GSM8K dataset."""
        try:
            logger.info("Loading GSM8K dataset...")
            self.dataset = load_dataset("gsm8k", "main", split=self.split)
            logger.info(f"Loaded {len(self.dataset)} examples from GSM8K {self.split} split")
        except Exception as e:
            logger.error(f"Error loading GSM8K dataset: {e}")
            raise
            
    def format_prompt(self, question: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
        """Format the prompt with few-shot examples if provided."""
        prompt = "Solve the following math problem. You must format your final answer as '#### [number]' at the very end.\n\n"
        
        if few_shot_examples:
            for example in few_shot_examples:
                prompt += f"Question: {example['question']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
                
        prompt += f"Question: {question}\nAnswer:"
        return prompt
        
    def extract_answer(self, response: Union[str, List[str]]) -> float:
        """Extract the final numerical answer from the model's response."""
        try:
            if isinstance(response, list):
                response = response[0] if response else ""
            
            import re
            
            # First attempt: extract strictly using the instructed "####" format
            if "####" in response:
                target = response.split("####")[-1].strip()
                clean_target = target.replace(',', '')
                matches = re.findall(r"[-+]?\d*\.?\d+", clean_target)
                if matches:
                    return float(matches[0])  # Take the first number found after ####
                    
            # Fallback: Clean commas (e.g., 2,125 -> 2125) and grab the last number
            clean_resp = str(response).replace(',', '')
            # Find all numbers including negatives and decimals
            matches = re.findall(r"[-+]?\d*\.?\d+", clean_resp)
            
            if matches:
                return float(matches[-1])  # Take the last number in the response
            return None
        except Exception as e:
            logger.warning(f"Error extracting answer from response: {e}")
            return None
            
    def evaluate_batch(self, model: BaseModel, examples: List[Dict], few_shot_examples: List[Dict]) -> List[Dict]:
        """Evaluate a batch of examples."""
        prompts = [self.format_prompt(ex["question"], few_shot_examples) for ex in examples]
        
        try:
            responses = model.generate(prompts, **self.generation_kwargs)
            batch_results = []
            
            for i, example in enumerate(examples):
                response = responses[i]
                predicted_answer = self.extract_answer(response)
                
                # Parse correct answer securely
                correct_str = example["answer"].split()[-1].replace(',', '')
                correct_answer = float(correct_str)
                
                is_correct = abs(predicted_answer - correct_answer) < 1e-6 if predicted_answer is not None else False
                
                if not is_correct:
                    msg = f"\n[INCORRECT] Expected: {correct_answer} | Predicted: {predicted_answer}\nModel Response:\n{response}\n{'-'*40}"
                    tqdm.write(msg)
                
                batch_results.append({
                    "question": example["question"],
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "response": response,
                    "is_correct": is_correct
                })
                
            return batch_results
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            return [{"question": ex["question"], "error": str(e)} for ex in examples]
            
    def run(self, model: BaseModel) -> Dict:
        """Run the GSM8K benchmark on the given model."""
        if self.dataset is None:
            self.load_dataset()
            
        results = []
        total_correct = 0
        total_examples = 0
        
        # Get few-shot examples
        few_shot_examples = self.dataset.select(range(self.num_shots))
        eval_dataset = self.dataset.select(range(self.num_shots, len(self.dataset)))
        
        if self.dataset_fraction < 1.0:
            eval_size = int(len(eval_dataset) * self.dataset_fraction)
            eval_dataset = eval_dataset.select(range(eval_size))
            logger.info(f"Using {self.dataset_fraction*100}% of the evaluation dataset ({eval_size} examples)")
            
        batch_size = self.config.get("batch_size", 4)
        
        # Create batches
        batches = [eval_dataset[i:i + batch_size] for i in range(0, len(eval_dataset), batch_size)]
        
        # Evaluate each batch
        for batch_dict in tqdm(batches, desc=f"Evaluating GSM8K (Batched by {batch_size})"):
            # Convert dictionary of lists into list of dictionaries
            batch = []
            num_items = len(batch_dict["question"])
            for idx in range(num_items):
                batch.append({
                    "question": batch_dict["question"][idx],
                    "answer": batch_dict["answer"][idx]
                })
                
            batch_results = self.evaluate_batch(model, batch, few_shot_examples)
            
            for result in batch_results:
                results.append(result)
                if "is_correct" in result:
                    total_examples += 1
                    if result["is_correct"]:
                        total_correct += 1
                        
        accuracy = total_correct / total_examples if total_examples > 0 else 0
        
        return {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "total_correct": total_correct,
            "detailed_results": results
        } 