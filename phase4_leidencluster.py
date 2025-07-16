
#Leiden clustering for multi-million embeddings  (GPU first, CPU fallback)

# • Uses FAISS-GPU if available; otherwise falls back to CPU index.
# • Vectorised batched k-NN search with tqdm progress bars.
# • Similarity cut keeps edges with cosine ≥ sim_cut.
# • Streams edges into igraph in constant-RAM chunks.
# • Runs Leiden CPM; deterministic with seed=42.


import json, time, argparse, pathlib
import numpy as np, faiss, igraph as ig, leidenalg
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────
def build_edges(index, vecs, k, sim_cut, batch):
    # Returns src,dst int32 arrays for an undirected k-NN graph.
    # Edges kept when inner-product ≥ sim_cut.

    n = vecs.shape[0]
    src_chunks, dst_chunks = [], []

    for beg in tqdm(range(0, n, batch), desc="FAISS batches"):
        end = min(beg + batch, n)
        D, I = index.search(vecs[beg:end], k + 1)     # D: inner-product
        src  = np.repeat(np.arange(beg, end, dtype=np.int32), k)
        dst  = I[:, 1:].astype(np.int32).reshape(-1)

        if sim_cut > 0:
            keep = D[:, 1:].reshape(-1) >= sim_cut
            src, dst = src[keep], dst[keep]

        keep = src < dst                              # one direction
        src_chunks.append(src[keep])
        dst_chunks.append(dst[keep])

    return np.concatenate(src_chunks), np.concatenate(dst_chunks)


def cluster_leiden_gpu(
        emb_path, id_path, out_path,
        k=50, sim_cut=0.25, resolution=1.0,
        batch=250_000):

    t0 = time.time()
    vecs = np.load(emb_path, mmap_mode="r")           # memory-map
    n, _ = vecs.shape
    ids  = json.load(open(id_path))
    assert len(ids) == n

    base = pathlib.Path(emb_path).stem.replace("embeddings_", "")
    cpu_index = faiss.read_index(
        str(pathlib.Path(emb_path).parent / f"{base}.index")
    )

    # Try GPU
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("⚡  FAISS-GPU detected → using GPU 0")
    except AttributeError:
        index = cpu_index
        print("faiss-gpu not found → falling back to CPU")

    # Build edges
    src, dst = build_edges(index, vecs, k, sim_cut, batch)
    print(f"🛠 edges kept: {len(src):,} "
          f"({len(src)/(n*k):.1%} of possible)")

    # Stream edges into igraph
    g = ig.Graph(n=n, directed=False)
    step = 5_000_000
    for i in tqdm(range(0, len(src), step), desc="igraph add_edges"):
        g.add_edges(list(zip(src[i:i+step], dst[i:i+step])))

    # Leiden
    print("Leiden clustering …")
    part = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=42
    )

    labels  = part.membership
    mapping = {pid: int(lbl) for pid, lbl in zip(ids, labels)}

    with open(out_path, "w") as f:
        json.dump(mapping, f)

    print(f"wrote {len(mapping)} labels → {out_path}  "
          f"(clusters={max(labels)+1}, time={time.time()-t0:.1f}s)")

# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--ids",        required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--k",          type=int,   default=50)
    ap.add_argument("--sim_cut",    type=float, default=0.25,
                    help="keep edges with cosine ≥ sim_cut")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--batch",      type=int,   default=250_000,
                    help="rows per FAISS search")
    args = ap.parse_args()

    cluster_leiden_gpu(
        emb_path=args.embeddings,
        id_path=args.ids,
        out_path=args.out,
        k=args.k,
        sim_cut=args.sim_cut,
        resolution=args.resolution,
        batch=args.batch
    )
