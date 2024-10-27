import pytest
from experiments.utils.evaluation import (
    extract_final_answer,
    majority_vote,
    check_answer_correctness
)

def test_extract_final_answer():
    # Test "therefore"
    text1 = "Some calculation. Therefore, the answer is 42 apples."
    assert extract_final_answer(text1) == "42"
    
    text2 = "Step 1: 10, Step 2: 20, Final step: 30"
    assert extract_final_answer(text2) == "30"
    
    assert extract_final_answer("") is None
    
    assert extract_final_answer("No numbers here") is None
    
    text3 = "Therefore, after spending 20 dollars, John has 50 dollars."
    assert extract_final_answer(text3) == "50"

def test_majority_vote():
    # Test majority voting with different scenarios
    # Test exact matches
    assert majority_vote(["42", "42", "42"]) == "42"
    
    # Test close numbers
    assert majority_vote(["42.1", "42", "42.05"]) == "42"
    
    # Test multiple groups
    assert majority_vote(["42", "42", "50", "51"]) == "42"
    
    # Test empty input
    assert majority_vote([]) is None
    
    # Test None values
    assert majority_vote([None, None]) is None
    assert majority_vote(["42", None, "42"]) == "42"

def test_check_answer_correctness():

    # Test exact matches
    assert check_answer_correctness("42", "42") == True
    
    # Test within tolerance
    assert check_answer_correctness("42.05", "42", tolerance=0.1) == True
    assert check_answer_correctness("42.2", "42", tolerance=0.1) == False
    
    # Test None handling
    assert check_answer_correctness(None, "42") == False
    