
#Phase-6  |  Turn candidate ID lists into publishable hypotheses.

import json, argparse, re, time, random, requests
from pathlib import Path
from tqdm import tqdm
import openai

SYSTEM_PROMPT = """
...
"""

DOT_NOVELTY_THRESHOLD = 0.90    # dot < 0.90 vs corpus method passes
MIN_DOIS = 2
MAX_RETRIES = 3

doi_rx = re.compile(r'10\.\d{4,9}/[^" ]+')

#DOI verifier
def valid_doi(doi, _cache={}):
    if doi in _cache: return _cache[doi]
    try:
        r = requests.head(f"https://doi.org/{doi}", timeout=4)
        ok = r.status_code in (303, 200)
    except requests.RequestException:
        ok = False
    _cache[doi] = ok
    return ok

# OpenAI call
def call_llm(problem_text, method_list, temperature):
    user_prompt = (
        f"Problem:\n{problem_text}\n\n"
        "Candidate methods:\n" +
        "\n".join(f"{i+1}) {m}" for i, m in enumerate(method_list)) +
        "\n\nTask: follow the instructions."
    )
    rsp = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ]
    )
    return rsp.choices[0].message.content

#Main pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="outputs/candidates.jsonl")
    ap.add_argument("--problems",   default="data/problems.jsonl")
    ap.add_argument("--methods",    default="data/methods.jsonl")
    ap.add_argument("--out",        default="outputs/hypotheses.jsonl")
    args = ap.parse_args()

    # load lookup tables
    problems = {r["problem_id"]: r["text"]
                for r in map(json.loads, open(args.problems))}
    methods  = {r["method_id"] : r["text"]
                for r in map(json.loads, open(args.methods))}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as fout, open(args.candidates) as fin:
        for line in tqdm(fin, desc="problems"):
            cand = json.loads(line)
            ptxt = problems[cand["problem_id"]]
            mtxts = [methods[mid] for mid in cand["candidate_method_ids"]]

            for attempt in range(MAX_RETRIES):
                temp = 0.7 + 0.1 * attempt
                try:
                    raw = call_llm(ptxt, mtxts, temp)
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue   # retry

                # basic validation
                dois = [d for d in data.get("citations", []) if valid_doi(d)]
                if len(dois) < MIN_DOIS:
                    continue   # retry

                # keep
                data.update(problem=ptxt, citations=dois)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                break

if __name__ == "__main__":
    main()
