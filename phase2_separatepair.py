# Extract separate problem / method tables from the existing dataset

# Input : dataset.jsonl
# Output: data/problems.jsonl
#         data/methods.jsonl
#         data/pairs.jsonl      # keeps original pairing & metadata


import json, pathlib, argparse, hashlib

parser = argparse.ArgumentParser()
parser.add_argument("--infile",  default="dataset.jsonl")
parser.add_argument("--outdir",  default="data", help="folder for three jsonl files")
args = parser.parse_args()

out_dir = pathlib.Path(args.outdir)
out_dir.mkdir(parents=True, exist_ok=True)

problems_f = (out_dir / "problems.jsonl").open("w")
methods_f  = (out_dir / "methods.jsonl").open("w")
pairs_f    = (out_dir / "pairs.jsonl").open("w")

def uid(text: str) -> str:
    "Stable SHA1 for deduplication / join-key later."
    return hashlib.sha1(text.encode()).hexdigest()[:16]

seen_problem, seen_method = set(), set()

with open(args.infile) as f:
    for line in f:
        rec = json.loads(line)

        problem = rec["problem"].strip()
        method  = rec["method"].strip()

        # 1) write pair record (keeps metadata)
        pair_rec = {
            "problem_id": uid(problem),
            "method_id" : uid(method),
            "problem"   : problem,
            "method"    : method,
            "paper_id"  : rec.get("corpusid")
        }
        pairs_f.write(json.dumps(pair_rec, ensure_ascii=False) + "\n")

        # 2) write unique problem table
        if problem not in seen_problem:
            seen_problem.add(problem)
            problems_f.write(json.dumps({
                "problem_id": uid(problem),
                "text": problem,
            }, ensure_ascii=False) + "\n")

        # 3) write unique method table
        if method not in seen_method:
            seen_method.add(method)
            methods_f.write(json.dumps({
                "method_id": uid(method),
                "text": method,
            }, ensure_ascii=False) + "\n")

print("Complete.  Problems:", len(seen_problem),
      "| Methods:", len(seen_method),
      "| Pairs:",   len(open(args.infile).read().splitlines()))