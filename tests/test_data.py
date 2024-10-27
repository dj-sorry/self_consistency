import pytest
from experiments.data.problems import get_few_shot_examples, load_addsum_examples

def test_few_shot_examples():

    examples = get_few_shot_examples()
    
    assert isinstance(examples, list)
    assert len(examples) > 0
    
    for example in examples:
        assert isinstance(example, dict)
        assert "question" in example
        assert "solution" in example
        assert isinstance(example["question"], str)
        assert isinstance(example["solution"], str)
        assert "Let's solve this step by step:" in example["solution"]
        assert "Therefore" in example["solution"]

def test_addsum_examples():

    problems = load_addsum_examples()

    assert isinstance(problems, list)
    assert len(problems) > 0
    
    for problem in problems:
        assert isinstance(problem, dict)
        assert "question" in problem
        assert "answer" in problem
        assert "solution" in problem
        
        assert isinstance(problem["question"], str)
        assert isinstance(problem["answer"], str)
        assert isinstance(problem["solution"], str)
        
        assert problem["solution"].count(")") >= 2  
        assert "Therefore" in problem["solution"]
        
        assert float(problem["answer"])

def test_answer_consistency():
    # Test that answers are consistent with solutions
    problems = load_addsum_examples()
    
    for problem in problems:
        import re
        numbers = re.findall(r"(\d+)", problem["solution"])
        final_number = numbers[-1]
        
        assert final_number == problem["answer"]