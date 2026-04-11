"""
Hybrid prompting method.

This method combines:
1) Auto-CoT style reasoning sketch
2) Least-to-Most style decomposition
3) One-pass self-refinement
"""

from models.loader import generate_response


HYBRID_SKETCH = """Create a compact plan for solving the following math problem.

Problem: {problem}

Requirements:
- Provide 3 to 5 short steps.
- Keep steps actionable and ordered.
- Do not compute final numbers here.

Output format:
Plan:
1) ...
2) ...
"""


HYBRID_DECOMPOSE_AND_SOLVE = """Solve the problem by following the plan and decomposing into sub-problems.

Problem: {problem}

Plan:
{plan}

Requirements:
- Break into 2 to 4 sub-problems implicitly while solving.
- Use concise reasoning (at most 8 short lines total).
- Use exact arithmetic; avoid unnecessary decimals.
- Do not output code blocks.
- Do not output multiple candidate answers.
- The final line must be exactly: Final Answer: <single integer or simplified fraction>

Output format (follow strictly):
1) Setup
2) Key equations
3) Compute
4) Quick check
Final Answer: <single integer or simplified fraction>
"""


HYBRID_REFINE = """Review and minimally refine the solution below.

Problem: {problem}

Draft solution:
{draft}

Requirements:
- Only fix concrete mistakes.
- Keep concise reasoning (at most 6 short lines).
- Keep a single final answer.
- The final line must be exactly: Final Answer: <single integer or simplified fraction>

Output format (follow strictly):
1) Error fix
2) Recompute
3) Quick check
Final Answer: <single integer or simplified fraction>
"""


def solve_hybrid(model, tokenizer, problem: str, max_new_tokens: int = 320) -> str:
    """Solve a problem using the hybrid prompting strategy."""
    sketch_prompt = HYBRID_SKETCH.format(problem=problem)
    plan = generate_response(
        model,
        tokenizer,
        sketch_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    solve_prompt = HYBRID_DECOMPOSE_AND_SOLVE.format(problem=problem, plan=plan)
    draft = generate_response(
        model,
        tokenizer,
        solve_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    refine_prompt = HYBRID_REFINE.format(problem=problem, draft=draft)
    refined = generate_response(
        model,
        tokenizer,
        refine_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return refined
