"""
Auto-CoT prompting method.

Reference:
Zhang Z, Zhang A, Li M, Smola A. Automatic Chain of Thought Prompting
in Large Language Models. ICLR 2023.
"""

from models.loader import generate_response


AUTO_COT_SKETCH = """Create a compact reasoning sketch for this math problem.

Problem: {problem}

Requirements:
- Provide only high-level steps (3 to 5 bullets).
- Do not compute final numbers here.
- Keep each bullet under one sentence.

Output format:
Reasoning Sketch:
1) ...
2) ...
"""


AUTO_COT_SOLVE = """Use the reasoning sketch to solve the problem.

Problem: {problem}

Reasoning Sketch:
{reasoning_sketch}

Requirements:
- Follow the sketch but fix any obvious issues.
- Use concise reasoning (at most 8 short lines total).
- Use exact arithmetic; avoid unnecessary decimals.
- Do not output code blocks.
- The final line must be exactly: Final Answer: <single integer or simplified fraction>

Output format (follow strictly):
1) Setup
2) Compute
3) Quick check
Final Answer: <single integer or simplified fraction>
"""


def solve_auto_cot(model, tokenizer, problem: str, max_new_tokens: int = 320) -> str:
    """
    Solve problem using Auto-CoT prompting.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        problem: Math problem text
        max_new_tokens: Maximum tokens to generate per stage

    Returns:
        Model response containing final solution and final answer
    """
    sketch_prompt = AUTO_COT_SKETCH.format(problem=problem)
    reasoning_sketch = generate_response(
        model,
        tokenizer,
        sketch_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    solve_prompt = AUTO_COT_SOLVE.format(
        problem=problem,
        reasoning_sketch=reasoning_sketch,
    )
    response = generate_response(
        model,
        tokenizer,
        solve_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return response
