# Phase 3 (dual): build separate FAISS indexes for problems and methods.


import json, argparse, pathlib, numpy as np, faiss, torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

torch.set_float32_matmul_precision("high")

# ─────────────── CLI ───────────────
p = argparse.ArgumentParser()
p.add_argument("--problems", default="data/problems.jsonl")
p.add_argument("--methods",  default="data/methods.jsonl")
p.add_argument("--outdir",   default="embeds")
p.add_argument("--model",    default="intfloat/e5-large-v2")
p.add_argument("--batch",    type=int, default=1024)
args = p.parse_args()

out = pathlib.Path(args.outdir); out.mkdir(exist_ok=True)

def load_texts(path, id_key):
    with open(path) as f:
        recs = [json.loads(l) for l in f]
    texts = [r["text"].strip() for r in recs]
    ids   = [r[id_key]          for r in recs]
    return texts, ids

print("Loading data …")
problem_txts, problem_ids = load_texts(args.problems, "problem_id")
method_txts,  method_ids  = load_texts(args.methods,  "method_id")

print("Loading model …")
model = SentenceTransformer(args.model, device="cuda")
model.max_seq_length = 256

def encode(texts, fname):
    all_vecs = []
    for i in tqdm(range(0, len(texts), args.batch), desc=f"Embedding {fname}"):
        batch = texts[i : i+args.batch]
        with torch.inference_mode():
            vecs = model.encode(batch,
                                batch_size=len(batch),
                                convert_to_numpy=True,
                                normalize_embeddings=True)
        all_vecs.append(vecs)
    vecs = np.vstack(all_vecs).astype("float32")
    np.save(out / f"{fname}.npy", vecs)
    print(f"saved {fname}.npy   shape={vecs.shape}")
    return vecs

prob_vecs = encode(problem_txts, "embeddings_problem")
meth_vecs = encode(method_txts,  "embeddings_method")

def build_index(vecs, fname):
    d = vecs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(vecs)
    index_path = out / f"{fname}.index"
    faiss.write_index(idx, str(index_path))
    print(f"saved {fname}.index   n={idx.ntotal}")

build_index(prob_vecs, "problem")
build_index(meth_vecs,  "method")

# optional: save row-order IDs for reverse lookup
json.dump(problem_ids, open(out / "problem_ids.json", "w"))
json.dump(method_ids,  open(out / "method_ids.json",  "w"))
print("Dual embeddings & indexes complete.")
