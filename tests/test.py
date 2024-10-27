import pytest
from experiments.src.experiment import ChainOfThoughtExperiment
from experiments.data.problems import load_addsum_examples

@pytest.fixture
def experiment():
    return ChainOfThoughtExperiment(model_name="google/flan-t5-base")

@pytest.fixture
def sample_problem():
    problems = load_addsum_examples()
    return problems[0]

def test_cot_prompt_generation(experiment, sample_problem):
    prompt = experiment.generate_cot_prompt(sample_problem["question"])
    
    assert "Question:" in prompt
    assert "Let's solve this step by step:" in prompt
    assert sample_problem["question"] in prompt

def test_reasoning_path_generation(experiment, sample_problem):
    paths = experiment.generate_reasoning_paths(
        sample_problem["question"],
        num_samples=2,
        temperature=0.7
    )
    
    assert len(paths) == 2
    assert all(isinstance(path, str) for path in paths)
    assert all(len(path) > 0 for path in paths)

def test_question_evaluation(experiment, sample_problem):
    result = experiment.evaluate_question(
        sample_problem["question"],
        sample_problem["answer"],
        num_samples=3
    )
    
    # Check result 
    assert isinstance(result, dict)
    assert "standard_correct" in result
    assert "sc_correct" in result
    assert "standard_path" in result
    assert "sc_paths" in result
    assert "standard_answer" in result
    assert "sc_answer" in result
    
    # check types
    assert isinstance(result["standard_correct"], bool)
    assert isinstance(result["sc_correct"], bool)
    assert isinstance(result["standard_path"], str)
    assert isinstance(result["sc_paths"], list)
    assert len(result["sc_paths"]) == 3

def test_model_initialization():
    # Test that different model types initialize corrrctly

    causal_exp = ChainOfThoughtExperiment(model_name="facebook/opt-1.3b")
    assert causal_exp.is_causal == True

    seq2seq_exp = ChainOfThoughtExperiment(model_name="google/flan-t5-base")
    assert seq2seq_exp.is_causal == False