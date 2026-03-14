"""
create_gt_topometric.py
Creates nodes_gt_topometric.pickle for a single episode from:
  - images_sem/*.npy  (GT semantic instance IDs from Habitat sim)
  - images/*.png      (RGB images, for spatial reference)

Run from the repo root:
  python create_gt_topometric.py ./data/hm3d_iin_val/4ok3usBNeis_0000000_chair_8_
"""

import sys
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from natsort import natsorted
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).parent))
from libs.common import utils


# ── helpers (mirrored from map_topo.py) ──────────────────────────────────────

def get_nbrs_delaunay(tri, v):
    nbrs = set()
    for simplex in tri.simplices:
        if v in simplex:
            nbrs.update(simplex)
    nbrs.discard(v)
    return [[v, n] for n in nbrs]


def remove_duplicate_nbr_pairs(nbrList):
    seen = set()
    unique = []
    for pair in nbrList:
        key = (min(pair), max(pair))
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def create_edges_delaunay(mask_cords):
    if len(mask_cords) > 3:
        tri = Delaunay(mask_cords)
        nbrs = []
        for v in range(len(mask_cords)):
            nbrs += get_nbrs_delaunay(tri, v)
        nbrs = remove_duplicate_nbr_pairs(nbrs)
    else:
        n = len(mask_cords)
        nbrs = [[u, v] for u in range(n) for v in range(u + 1, n)]
    return np.array(nbrs) if nbrs else np.empty((0, 2), dtype=int)


# ── main ──────────────────────────────────────────────────────────────────────

def build_gt_topometric_graph(episode_dir: Path, out_name="nodes_gt_topometric.pickle"):
    sem_dir = episode_dir / "images_sem"
    sem_paths = natsorted(sem_dir.glob("*.npy"))
    assert len(sem_paths) > 0, f"No semantic masks found in {sem_dir}"

    G = nx.Graph()
    node_id = 0
    prev_last_node = None

    for img_idx, sem_path in enumerate(sem_paths):
        sem = np.load(sem_path, allow_pickle=True)  # (H, W) uint16 instance IDs

        unique_ids = np.unique(sem)
        unique_ids = unique_ids[unique_ids > 0]  # skip background (0)

        masks = []
        instance_ids = []
        for inst_id in unique_ids:
            mask = (sem == inst_id)
            if mask.sum() < 10:   # skip tiny masks
                continue
            masks.append(mask)
            instance_ids.append(int(inst_id))

        if len(masks) == 0:
            continue

        # mask centroids (x, y) for Delaunay
        centroids = np.array([
            np.array(np.nonzero(m)).mean(1)[::-1]   # (col, row) = (x, y)
            for m in masks
        ])

        # intra-image edges via Delaunay
        nbrs = create_edges_delaunay(centroids)

        first_node_in_img = node_id
        nodes = []
        for j, (mask, inst_id) in enumerate(zip(masks, instance_ids)):
            rle = utils.mask_to_rle_numpy(mask[None, ...])[0]
            nodes.append((node_id, {
                "map":         [img_idx, j],
                "instance_id": inst_id,
                "segmentation": rle,
            }))
            node_id += 1

        edges = []
        for u, v in nbrs:
            edges.append((first_node_in_img + int(u),
                          first_node_in_img + int(v),
                          {"margin": 0.0, "edgeType": "intra"}))

        # temporal edge connecting last node of previous image to first of this one
        if prev_last_node is not None and len(nodes) > 0:
            edges.append((prev_last_node, first_node_in_img,
                          {"margin": 0.0, "edgeType": "temporal"}))

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        if len(nodes) > 0:
            prev_last_node = first_node_in_img + len(nodes) - 1

    # inter-image edges: connect nodes with the same instance_id across frames
    # (this is the "GT data association" that replaces LightGlue matching)
    instid_to_nodes: dict[int, list] = {}
    for n, data in G.nodes(data=True):
        iid = data["instance_id"]
        instid_to_nodes.setdefault(iid, []).append(n)

    da_edges = []
    for iid, node_list in instid_to_nodes.items():
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                u, v = node_list[i], node_list[j]
                # only connect nodes from nearby frames (window = 5)
                img_u = G.nodes[u]["map"][0]
                img_v = G.nodes[v]["map"][0]
                if abs(img_u - img_v) <= 5:
                    da_edges.append((u, v, {"margin": 0.0, "edgeType": "da"}))

    G.add_edges_from(da_edges)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    out_path = episode_dir / out_name
    pickle.dump(G, open(out_path, "wb"))
    print(f"Saved to {out_path}")
    return G


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    build_gt_topometric_graph(episode_dir)