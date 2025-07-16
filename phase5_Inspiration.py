#Phase 5: Inspiration Mining (hybrid B→C→A cross-search + compatibility filter)
import json
import numpy as np
import faiss
import pickle
from collections import defaultdict


EMBED_DIR = "embeds"
DATA_DIR  = ""
CLUSTER_DIR = "clusters"
COMPAT_CLF = "compatibility_clf.pkl"

# retrieval sizes
K_PROB_NEIGHBORS = 10  # k1
METHODS_PER_NEIGHBOR = 5  # m
K_CLUSTER_BRIDGE = 20
K_DIRECT = 5

COMPAT_THRESHOLD = 0.4

# Loading index & vector from pre-generated files
print("Loading indexes & embeddings…")
problem_index = faiss.read_index(f"{EMBED_DIR}/problem.index")
method_index  = faiss.read_index(f"{EMBED_DIR}/method.index")

problem_vecs = np.load(f"{EMBED_DIR}/embeddings_problem.npy")
method_vecs  = np.load(f"{EMBED_DIR}/embeddings_method.npy")

# Loading ID and problem/method mapping
print("Loading ID mappings…")
with open(f"{EMBED_DIR}/problem_ids.json") as f:
    problem_ids = json.load(f)
with open(f"{EMBED_DIR}/method_ids.json")  as f:
    method_ids  = json.load(f)

# invert for lookup: id → row
problem_id_to_idx = {pid: i for i, pid in enumerate(problem_ids)}
method_id_to_idx  = {mid: i for i, mid in enumerate(method_ids)}

# Loading original problem-method pairs
print("Loading original pairs…")
pairs_by_problem = defaultdict(list)
with open(f"{DATA_DIR}/pairs.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        pairs_by_problem[rec["problem_id"]].append(rec["method_id"])

# Loading cluster labels (optional)
try:
    with open(f"{CLUSTER_DIR}/problem_clusters.json") as f:
        problem_clusters = json.load(f)
    with open(f"{CLUSTER_DIR}/method_clusters.json")  as f:
        method_clusters  = json.load(f)
    use_clusters = True
    print("Cluster labels loaded.")
except FileNotFoundError:
    use_clusters = False
    print("No cluster labels found; skipping variant C.")

# Loading compatibility classifier
print("Loading compatibility classifier…")
with open(COMPAT_CLF, "rb") as f:
    compat_clf = pickle.load(f)

# Also load raw text for compatibility prompt
problems_text = { rec["problem_id"]: rec["text"]
                  for rec in json.load(open(f"{DATA_DIR}/problems.jsonl")) }
methods_text  = { rec["method_id"] : rec["text"]
                  for rec in json.load(open(f"{DATA_DIR}/methods.jsonl")) }

# Retrieval Variants

# A) Direct Knn cluster search
def direct_knn_candidates(p_vec, k=K_DIRECT):
    D, I = method_index.search(p_vec.reshape(1, -1), k)
    return [ method_ids[i] for i in I[0] ]

# B) Two hop search
def two_hop_candidates(p_id, k1=K_PROB_NEIGHBORS, m=METHODS_PER_NEIGHBOR):
    idxs = problem_index.search(
        problem_vecs[problem_id_to_idx[p_id]].reshape(1,-1), k1
    )[1][0]
    cands = set()
    for rid in idxs:
        pid = problem_ids[rid]
        for mid in pairs_by_problem.get(pid, [])[:m]:
            cands.add(mid)
    return list(cands)

# C) Cluster bridge search
def cluster_bridge_candidates(p_id, k=K_CLUSTER_BRIDGE):
    if not use_clusters:
        return []
    p_cluster = problem_clusters[p_id]
    D, I = method_index.search(
        problem_vecs[problem_id_to_idx[p_id]].reshape(1,-1), k
    )
    cands = []
    for rid in I[0]:
        mid = method_ids[rid]
        if method_clusters.get(mid) != p_cluster:
            cands.append(mid)
    return cands


# checks if existing pair was matched
def compatibility_filter(candidate_ids):
    prompts = [
        f"{problems_text[target_problem_id]} [SEP] {methods_text[mid]}"
        for mid in candidate_ids
    ]
    probs  = compat_clf.predict_proba(prompts)[:, 1]
    return [
        mid for mid, p in zip(candidate_ids, probs)
        if p >= COMPAT_THRESHOLD
    ]

# Main function
def inspire(target_problem_id):
    p_idx = problem_id_to_idx[target_problem_id]
    p_vec  = problem_vecs[p_idx]

    # B → two-hop
    cand_B = set(two_hop_candidates(target_problem_id))

    # C → cluster bridge
    cand_C = set(cluster_bridge_candidates(target_problem_id))

    # A → direct KNN fallback
    cand_A = set(direct_knn_candidates(p_vec))

    # union & filter
    all_cands = cand_B | cand_C | cand_A
    filtered  = compatibility_filter(list(all_cands))
    return filtered

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("problem_id", help="target problem_id to generate candidates for")
    args = p.parse_args()
    target_problem_id = args.problem_id

    print(f"Inspiring methods for problem {target_problem_id} …")
    methods = inspire(target_problem_id)
    for mid in methods:
        print(f" - {mid}: {methods_text[mid]}")
