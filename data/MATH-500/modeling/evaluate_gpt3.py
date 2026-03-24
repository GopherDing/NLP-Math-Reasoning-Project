"""
GPT-3 Evaluation Script for MATH dataset.

This file is from the original MATH project (hendrycks/competition_math).
It uses the deprecated OpenAI Completion API.

NOTE: This script is kept for historical reference. For modern evaluations,
use the OpenAI Chat API or the project's own evaluation/metrics.py instead.

Set your API key via environment variable:
    export OPENAI_API_KEY="sk-..."
    set OPENAI_API_KEY="sk-..."
"""

import os
import re
import json
import sys

# Resolve modeling directory
_MOD_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to import the MATH project's own modules
sys.path.insert(0, _MOD_DIR)
try:
    from dataset.util import last_boxed_only_string
    from math_equivalence import is_equiv
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False
    last_boxed_only_string = None
    is_equiv = None


def remove_boxed(s):
    """Remove \\boxed{...} wrapper from answer string."""
    if not s:
        return None
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except (AssertionError, IndexError):
        return None


TRAIN_PROMPT = (
    "Given a mathematics problem, determine the answer. Simplify your answer as much as possible."
    + "\n" + "Problem: What is $\\left(\\frac{7}{8}\\right)^3 \\cdot \\left(\\frac{7}{8}\\right)^{-3}$?"
    + "\n" + "Answer: $1$"
    + "\n" + "###"
    + "\n" + "Problem: In how many ways can 4 books be selected from a shelf of 6 books if the order in which the books are selected does not matter?"
    + "\n" + "Answer: $15$"
    + "\n" + "###"
    + "\n" + "Problem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$"
    + "\n" + "Answer: $\\sqrt{59}$"
    + "\n" + "###"
    + "\n" + "Problem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?"
    + "\n" + "Answer: $\\frac{1}{32}$"
    + "\n" + "###"
    + "\n" + "Problem: The first three terms of an arithmetic sequence are 1, 10 and 19, respectively. What is the value of the 21st term?"
    + "\n" + "Answer: $181$"
    + "\n" + "###"
    + "\n" + "Problem: Calculate $6 \\cdot 8\\frac{1}{3}"
    + "\n" + "Answer: $50$"
    + "\n" + "###"
    + "\n" + "Problem: When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
    + "\n" + "Answer: $2$"
    + "\n" + "###"
    + "\n" + "Problem: How many zeros are at the end of the product 25 $\\times$ 240?"
    + "\n" + "Answer: $3$"
    + "\n" + "###"
)

ROOTDIR = os.path.join(os.path.dirname(_MOD_DIR), "MATH", "data", "test")


def _get_answer_from_solution(solution_text):
    """Extract boxed answer from solution."""
    if not solution_text:
        return None
    boxed_match = re.search(r'\\boxed\{([^{}]+)\}', solution_text)
    if boxed_match:
        return boxed_match.group(1).strip()
    return None


def call_engine(train_prompt, problem, engine="davinci"):
    """
    Given a problem, returns the most likely answer determined by the GPT engine.
    Uses the deprecated OpenAI Completion API.
    """
    import openai
    import operator as _op

    openai.api_key = os.environ.get("OPENAI_API_KEY", "")

    test_question = "\n" + problem + "\n" + "Answer: $"
    prompt = train_prompt + test_question
    num_tokens = 20
    c = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=num_tokens,
        logprobs=100,
        temperature=0,
        echo=True
    )
    tokens = c["choices"][0]["logprobs"]["tokens"]
    startindex = -1 * num_tokens
    endindex = -1 * num_tokens + 1
    for token in tokens[startindex + 1:]:
        if token == "$" or token == "###" or token == "\n":
            break
        else:
            endindex += 1
    final_answer = ""
    for i in range(startindex, endindex):
        all_answers = c["choices"][0]["logprobs"]["top_logprobs"][i]
        best_answer = max(all_answers.items(), key=_op.itemgetter(1))[0]
        final_answer += best_answer
    return final_answer


def run(engine="davinci", max_samples=-1):
    """
    Run GPT evaluation on MATH test set.

    Args:
        engine: OpenAI engine name (e.g. "davinci", "curie")
        max_samples: Max number of samples to process (-1 for all)
    """
    if not _IMPORT_OK:
        raise RuntimeError(
            "Cannot run: missing dependencies (numpy, dataset.util, math_equivalence). "
            "These are optional dependencies from the original MATH project."
        )

    import numpy as np

    outputs = []
    answers = []
    types_list = []
    levels = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0

    for subdir, dirs, files in os.walk(ROOTDIR):
        for file in files:
            fnames_list.append(os.path.join(subdir, file))
            with open(os.path.join(subdir, file), 'r') as fp:
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {file}: {e}")
                    raise

                prob_level = problem_data.get("level", "")
                prob_type = problem_data.get("type", "")
                try:
                    prob_level = int(prob_level.split("Level ")[1])
                except (ValueError, IndexError, AttributeError):
                    prob_level = None

                model_output = call_engine(TRAIN_PROMPT, problem_data["problem"], engine=engine)

                # Extract ground truth answer
                solution_text = problem_data.get("solution", "")
                if last_boxed_only_string:
                    answer = remove_boxed(last_boxed_only_string(solution_text))
                else:
                    answer = _get_answer_from_solution(solution_text)

                levels.append(prob_level)
                types_list.append(prob_type)
                outputs.append(model_output)
                answers.append(answer)

                print("Model output:", model_output)
                print("Correct answer:", answer)
                print("-" * 40)

                try:
                    equiv = is_equiv(model_output, answer)
                except Exception:
                    equiv = False

                if (prob_level, prob_type) in cors:
                    cors[(prob_level, prob_type)].append(equiv)
                else:
                    cors[(prob_level, prob_type)] = [equiv]

                if prob_level is not None:
                    if prob_level in level_cors:
                        level_cors[prob_level].append(equiv)
                    else:
                        level_cors[prob_level] = [equiv]

                if prob_type:
                    if prob_type in subject_cors:
                        subject_cors[prob_type].append(equiv)
                    else:
                        subject_cors[prob_type] = [equiv]

                if equiv:
                    correct += 1
                total += 1

                print(f"{correct}/{total}")

            if max_samples > 0 and total >= max_samples:
                break
        if max_samples > 0 and total >= max_samples:
            break

    # Write results
    out_fname = f"outputs_answers_gpt3_{engine}.txt"
    with open(out_fname, "w+", encoding="utf-8") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(
                zip(outputs, answers, types_list, levels, fnames_list)):
            f.write(f"{k} TYPE: {prob_type} | LEVEL: {prob_level} | OUTPUT: {output} | ANSWER: {answer} | FNAME: {fname}\n")

        f.write("#####################\n")
        subjects = ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability',
                    'Geometry', 'Intermediate Algebra', 'Precalculus']
        for subject in subjects:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors:
                    continue
                cors_list = cors[key]
                line = f"{subject} Level {level} Accuracy = {np.sum(cors_list)}/{len(cors_list)} = {np.mean(cors_list):.3f}"
                print(line)
                f.write(line + "\n")

        f.write("#####################\n")
        for level in sorted(level_cors):
            cors_list = level_cors[level]
            line = f"Level {level} Accuracy = {np.sum(cors_list)}/{len(cors_list)} = {np.mean(cors_list):.3f}"
            print(line)
            f.write(line + "\n")

        f.write("#####################\n")
        for subject in subjects:
            if subject not in subject_cors:
                continue
            cors_list = subject_cors[subject]
            line = f"{subject} Accuracy = {np.sum(cors_list)}/{len(cors_list)} = {np.mean(cors_list):.3f}"
            print(line)
            f.write(line + "\n")

        f.write("#####################\n")
        line = f"Overall Accuracy = {correct}/{total} = {correct/total:.3f}"
        print(line)
        f.write(line + "\n")


if __name__ == "__main__":
    engines = ["davinci", "curie", "babbage", "ada"][::-1]
    for engine in engines:
        run(engine)

    # For testing a small sample:
    # for engine in ["ada"]:
    #     run(engine, max_samples=10)
