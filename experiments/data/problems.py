from typing import List, Dict

"""
Ideally, test it on benchmarks like gsm8k, reclor, and so on.
These examples are just for educational purposes
"""

def get_few_shot_examples() -> List[Dict[str, str]]:
    return [
        {
            "question": "Joan found 70 seashells on the beach. She gave Sam some of her seashells. She has 27 seashells. How many seashells did she give to Sam?",
            "solution": """Let's solve this step by step:
1) Joan started with 70 seashells
2) She now has 27 seashells
3) To find how many she gave away: 70 - 27 = 43 seashells

Therefore, Joan gave Sam 43 seashells."""
        },
        {
            "question": "There were 28 bales of hay in the barn. Luke put in 23 more bales and Alex added 15 more. How many bales are in the barn now?",
            "solution": """Let's solve this step by step:
1) Started with 28 bales
2) Add Luke's bales: 28 + 23 = 51 bales
3) Add Alex's bales: 51 + 15 = 66 bales

Therefore, there are 66 bales in the barn now."""
        }
    ]

def load_addsum_examples() -> List[Dict[str, str]]:
    return [
        {
            "question": "There are 7 cards on the table. James took 3 cards. How many cards are still on the table?",
            "answer": "4",
            "solution": """1) Initially, there are 7 cards on the table
2) James took 3 cards away
3) To find remaining cards: 7 - 3 = 4

Therefore, there are 4 cards still on the table."""
        },
        {
            "question": "John has 5 marbles. He found 3 more marbles at the park and his friend gave him 4 marbles. How many marbles does John have now?",
            "answer": "12",
            "solution": """1) Initially, John has 5 marbles
2) He found 3 more marbles: 5 + 3 = 8 marbles
3) His friend gave him 4 more: 8 + 4 = 12 marbles

Therefore, John has 12 marbles now."""
        },
        {
            "question": "Sara had 8 dollars. She spent 3 dollars on lunch and then earned 5 dollars helping her neighbor. How many dollars does Sara have now?",
            "answer": "10",
            "solution": """1) Sara started with 8 dollars
2) After spending 3 dollars: 8 - 3 = 5 dollars
3) After earning 5 more: 5 + 5 = 10 dollars

Therefore, Sara has 10 dollars now."""
        }
    ]