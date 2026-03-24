# Prompt Templates for Mathematical Reasoning

## Chain of Thought (CoT)

COT_TEMPLATE = """Solve the following math problem step by step.

Problem: {problem}

Let's think through this step by step:
"""

## Self-Refine

SELF_REFINE_INITIAL = """Solve the following math problem:

Problem: {problem}

Solution:
"""

SELF_REFINE_FEEDBACK = """Review your solution to the following problem:

Problem: {problem}

Your solution:
{solution}

Please identify any errors or areas for improvement, then provide a refined solution.

Feedback and Refined Solution:
"""

## Self-Consistency

SELF_CONSISTENCY_TEMPLATE = """Solve the following math problem:

Problem: {problem}

Solution:
"""

## Least-to-Most (optional)

LEAST_TO_MOST_DECOMPOSE = """Break down the following problem into simpler sub-problems:

Problem: {problem}

Sub-problems:
1. 
"""

LEAST_TO_MOST_SOLVE = """Now solve each sub-problem:

{sub_problems}

Final answer:
"""
