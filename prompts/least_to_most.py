"""
Least-to-Most prompting method.

Reference:
Zhou D, Scharli N, Hou L, et al. Least-to-Most Prompting Enables
Complex Reasoning in Large Language Models. ICLR 2023.
"""

from models.loader import generate_response


LEAST_TO_MOST_DECOMPOSE = """Break the following math problem into a short sequence of sub-problems.

Problem: {problem}

Requirements:
- Provide 2 to 4 sub-problems only.
- Keep each sub-problem short and actionable.
- Focus on dependency order (easy to hard).
- Do not solve yet.

Output format:
Sub-problems:
1) ...
2) ...
"""


LEAST_TO_MOST_SOLVE = """Solve the math problem by following the sub-problems below.

Problem: {problem}

Sub-problems:
{sub_problems}

Requirements:
- Solve in the listed order.
- Use concise reasoning (at most 8 short lines total).
- Use exact arithmetic; avoid unnecessary decimals.
- Do not output code blocks.
- The final line must be exactly: Final Answer: <single integer or simplified fraction>

Output format (follow strictly):
1) Solve sub-problem 1
2) Solve sub-problem 2
3) Combine results
4) Quick check
Final Answer: <single integer or simplified fraction>
"""


def solve_least_to_most(model, tokenizer, problem: str, max_new_tokens: int = 320) -> str:
    """
    Solve problem using Least-to-Most prompting.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        problem: Math problem text
        max_new_tokens: Maximum tokens to generate per stage

    Returns:
        Model response containing final solution and final answer
    """
    decompose_prompt = LEAST_TO_MOST_DECOMPOSE.format(problem=problem)
    sub_problems = generate_response(
        model,
        tokenizer,
        decompose_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    solve_prompt = LEAST_TO_MOST_SOLVE.format(
        problem=problem,
        sub_problems=sub_problems,
    )
    response = generate_response(
        model,
        tokenizer,
        solve_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return response
