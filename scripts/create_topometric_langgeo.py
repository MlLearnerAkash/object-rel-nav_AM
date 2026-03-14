"""
create_topometric_langgeo.py

Builds a topometric graph (nodes_langgeo_topometric.pickle) for a single episode
directory using LangGeoNetPredictor for per-object goal-distance predictions.

Inputs read from episode_dir:
  instruction.txt      – navigation instruction string
  images/*.png         – RGB frames   (H × W × 3)
  images_sem/*.npy     – GT semantic instance maps  (H × W, uint16 instance IDs)

Node attributes (matching nodes_gt_topometric.pickle schema):
  map            : [frame_idx, obj_idx_in_frame]
  instance_id    : semantic instance ID from the simulator
  segmentation   : RLE-encoded binary mask  {"size": [H,W], "counts": [...]}
  area           : pixel count of the mask
  bbox           : (row_min, col_min, height, width)
  coords         : centroid as np.array([row, col])
  pred_distance  : goal distance predicted by LangGeoNet  (float)

Edge attributes:
  pred_distance_avg  : mean predicted distance of the two endpoint nodes
  edgeType           : "intra" | "temporal" | "da"

Graph-level attributes:
  costmaps       : list of per-frame costmaps  (H × W float32 ndarray)
  instruction    : the instruction string
  cfg            : {"episode_dir": str}

Usage (from repo root):
  python scripts/create_topometric_langgeo.py \\
      ./data/hm3d_iin_val/4ok3usBNeis_0000000_chair_8_ \\
      ./model_weights/latest.pth

  # custom output name
  python scripts/create_topometric_langgeo.py \\
      ./data/hm3d_iin_val/4ok3usBNeis_0000000_chair_8_ \\
      ./model_weights/latest.pth \\
      --out nodes_langgeo_topometric.pickle
"""

import sys
import argparse
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from PIL import Image
from natsort import natsorted
from scipy.spatial import Delaunay

import torch
from transformers import CLIPProcessor

# ── path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT.parent / "VLN-CE" / "costmap_predictor"))

from libs.common import utils          # mask_to_rle_numpy
from langgeonet.model import LangGeoNet


# ── LangGeoNetPredictor (adapted from VLN-CE/costmap_predictor reference) ────

class LangGeoNetPredictor:
    """Inference wrapper: preprocessing → prediction → costmap generation."""

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        self.model = LangGeoNet(
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            clip_model_name=cfg["clip_model"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.clip_processor = CLIPProcessor.from_pretrained(cfg["clip_model"])

        print(
            f"[LangGeoNetPredictor] Loaded epoch {ckpt['epoch']} "
            f"(val MAE={ckpt.get('best_val_mae', '?'):.4f}) on {self.device}"
        )

    @torch.no_grad()
    def predict_frame(self, image, masks, instruction):
        """
        Args:
            image:       PIL.Image or np.ndarray [H, W, 3]
            masks:       np.ndarray [K, H, W]  bool / uint8 per-instance masks
            instruction: str

        Returns:
            distances : np.ndarray [K_valid]  – predicted goal distances
            costmap   : np.ndarray [H, W]     – distance value per pixel
            valid_idx : np.ndarray [K_valid]  – original mask indices kept
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        enc = self.clip_processor(
            images=image,
            text=instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        pixel_values   = enc["pixel_values"].to(self.device)   # [1, 3, 224, 224]
        input_ids      = enc["input_ids"].to(self.device)       # [1, 77]
        attention_mask = enc["attention_mask"].to(self.device)  # [1, 77]

        valid_idx = np.where(masks.any(axis=(1, 2)))[0]
        valid_masks = masks[valid_idx]                           # [K_valid, H, W]

        H, W = masks.shape[1], masks.shape[2]
        if valid_masks.shape[0] == 0:
            return np.array([]), np.ones((H, W), dtype=np.float32), valid_idx

        masks_t = torch.from_numpy(valid_masks.astype(bool)).to(self.device)

        predictions, _ = self.model(
            images=pixel_values,
            masks_list=[masks_t],
            class_ids_list=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        distances = predictions[0].cpu().numpy()   # [K_valid]

        costmap = np.ones((H, W), dtype=np.float32)
        for k, orig_k in enumerate(valid_idx):
            costmap[valid_masks[k] > 0] = distances[k]

        return distances, costmap, valid_idx


# ── Edge attribute helpers ────────────────────────────────────────────────────

def _edge_attrs(distance: float, edge_type: str) -> dict:
    """
    Return an edge-attribute dict compatible with the GT topometric schema.
    All standard weight keys ('e3d', 'geodesic_min/avg/max') are populated
    with the same predicted distance value so the planner can use any of them
    regardless of which edge_weight_str is configured.
    """
    return {
        "e3d":           distance,
        "geodesic_min":  distance,
        "geodesic_avg":  distance,
        "geodesic_max":  distance,
        "pred_distance": distance,   # kept for traceability
        "edgeType":      edge_type,
    }


# ── Delaunay helpers ──────────────────────────────────────────────────────────

def _nbrs_delaunay(tri, v):
    nbrs = set()
    for simplex in tri.simplices:
        if v in simplex:
            nbrs.update(simplex)
    nbrs.discard(v)
    return [[v, n] for n in nbrs]


def _dedup_pairs(pairs):
    seen, unique = set(), []
    for p in pairs:
        key = (min(p), max(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _delaunay_edges(centroids):
    n = len(centroids)
    if n > 3:
        tri = Delaunay(centroids)
        pairs = []
        for v in range(n):
            pairs += _nbrs_delaunay(tri, v)
        pairs = _dedup_pairs(pairs)
    else:
        pairs = [[u, v] for u in range(n) for v in range(u + 1, n)]
    return np.array(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)


# ── build graph ───────────────────────────────────────────────────────────────

def build_langgeo_topometric_graph(
    episode_dir: Path,
    checkpoint_path: str,
    out_name: str = "nodes_langgeo_topometric.pickle",
    min_mask_pixels: int = 10,
    da_window: int = 5,
):
    """
    Build a topometric graph enriched with per-node LangGeoNet goal-distance
    predictions.

    Parameters
    ----------
    episode_dir       : path to one episode directory
    checkpoint_path   : path to LangGeoNet .pth checkpoint
    out_name          : output pickle filename (saved inside episode_dir)
    min_mask_pixels   : discard masks smaller than this
    da_window         : max frame-gap for data-association cross-frame edges
    """
    episode_dir = Path(episode_dir)

    # ── instruction ──────────────────────────────────────────────────────────
    instruction_path = episode_dir / "instruction.txt"
    assert instruction_path.exists(), f"instruction.txt not found in {episode_dir}"
    instruction = instruction_path.read_text().strip()
    print(f"[Episode] {episode_dir.name}")
    print(f"[Instruction] {instruction[:80]}...")

    # ── frame paths ──────────────────────────────────────────────────────────
    rgb_paths = natsorted((episode_dir / "images").glob("*.png"))
    sem_paths = natsorted((episode_dir / "images_sem").glob("*.npy"))
    assert len(rgb_paths) > 0, f"No RGB images found in {episode_dir / 'images'}"
    assert len(rgb_paths) == len(sem_paths), (
        f"RGB ({len(rgb_paths)}) and semantic ({len(sem_paths)}) frame counts differ"
    )
    print(f"[Frames] {len(rgb_paths)} frames")

    # ── predictor ────────────────────────────────────────────────────────────
    predictor = LangGeoNetPredictor(checkpoint_path)

    # ── build graph ──────────────────────────────────────────────────────────
    G = nx.Graph()
    node_id = 0
    prev_last_node = None    # last node of the previous frame (for temporal edges)
    costmaps = []            # one per frame
    temporal_edges = []      # stored in graph attrs (same convention as GT)

    for img_idx, (rgb_path, sem_path) in enumerate(zip(rgb_paths, sem_paths)):
        image = Image.open(rgb_path).convert("RGB")
        sem   = np.load(sem_path, allow_pickle=True)   # (H, W) uint16

        unique_ids = np.unique(sem)
        unique_ids = unique_ids[unique_ids > 0]        # exclude background

        # ── per-instance masks ───────────────────────────────────────────────
        raw_masks    = []
        instance_ids = []
        for inst_id in unique_ids:
            mask = (sem == inst_id)
            if mask.sum() < min_mask_pixels:
                continue
            raw_masks.append(mask)
            instance_ids.append(int(inst_id))

        H, W = sem.shape
        if len(raw_masks) == 0:
            costmaps.append(np.ones((H, W), dtype=np.float32))
            continue

        masks_arr = np.stack(raw_masks, axis=0).astype(np.uint8)  # [K, H, W]

        # ── LangGeoNet inference ─────────────────────────────────────────────
        distances, costmap, valid_idx = predictor.predict_frame(
            image=np.array(image), masks=masks_arr, instruction=instruction
        )
        costmaps.append(costmap)

        # Build a mapping: original index → predicted distance
        # For masks that were filtered as empty by the predictor, use np.nan
        dist_per_mask = np.full(len(raw_masks), np.nan, dtype=np.float32)
        for k, orig_k in enumerate(valid_idx):
            dist_per_mask[orig_k] = distances[k]

        # ── centroids for Delaunay ───────────────────────────────────────────
        centroids = np.array([
            np.array(np.nonzero(m)).mean(axis=1)[::-1]   # (col, row) = (x, y)
            for m in raw_masks
        ])

        intra_pairs = _delaunay_edges(centroids)

        # ── nodes ────────────────────────────────────────────────────────────
        first_node_in_frame = node_id
        nodes = []
        for j, (mask, inst_id, dist) in enumerate(
            zip(raw_masks, instance_ids, dist_per_mask)
        ):
            rows, cols = np.nonzero(mask)
            r_min, r_max = int(rows.min()), int(rows.max())
            c_min, c_max = int(cols.min()), int(cols.max())
            centroid_rc  = np.array([rows.mean(), cols.mean()])

            rle = utils.mask_to_rle_numpy(mask[None, ...])[0]

            nodes.append((node_id, {
                "map":           [img_idx, j],
                "instance_id":   inst_id,
                "segmentation":  rle,
                "area":          int(mask.sum()),
                "bbox":          (r_min, c_min, r_max - r_min + 1, c_max - c_min + 1),
                "coords":        centroid_rc,
                "pred_distance": float(dist) if not np.isnan(dist) else None,
            }))
            node_id += 1

        # ── intra-frame edges (Delaunay) ─────────────────────────────────────
        intra_edges = []
        for u, v in intra_pairs:
            nu = first_node_in_frame + int(u)
            nv = first_node_in_frame + int(v)
            d_avg = float(np.nanmean([dist_per_mask[u], dist_per_mask[v]]))
            intra_edges.append((nu, nv, _edge_attrs(d_avg, "intra")))

        # ── temporal edge (last node of prev frame → first of this frame) ────
        if prev_last_node is not None and len(nodes) > 0:
            temporal_edges.append([
                prev_last_node,
                first_node_in_frame,
                {"sim": 0},
            ])

        G.add_nodes_from(nodes)
        G.add_edges_from(intra_edges)

        if len(nodes) > 0:
            prev_last_node = first_node_in_frame + len(nodes) - 1

        if (img_idx + 1) % 10 == 0 or img_idx == len(rgb_paths) - 1:
            print(f"  frame {img_idx + 1}/{len(rgb_paths)}: "
                  f"{len(nodes)} nodes, running total {node_id}")

    # ── temporal edges ────────────────────────────────────────────────────────
    for u, v, attrs in temporal_edges:
        G.add_edge(u, v, **_edge_attrs(0.0, "temporal"))

    # ── data-association cross-frame edges (same instance_id, nearby frames) ─
    instid_to_nodes: dict[int, list] = {}
    for n, data in G.nodes(data=True):
        instid_to_nodes.setdefault(data["instance_id"], []).append(n)

    da_edges = []
    for node_list in instid_to_nodes.values():
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                u, v = node_list[i], node_list[j]
                frame_u = G.nodes[u]["map"][0]
                frame_v = G.nodes[v]["map"][0]
                if abs(frame_u - frame_v) <= da_window:
                    d_u = G.nodes[u]["pred_distance"]
                    d_v = G.nodes[v]["pred_distance"]
                    vals = [x for x in [d_u, d_v] if x is not None]
                    d_avg = float(np.mean(vals)) if vals else 0.0
                    da_edges.append((u, v, _edge_attrs(d_avg, "da")))
    G.add_edges_from(da_edges)

    # ── graph-level attributes ────────────────────────────────────────────────
    G.graph["temporalEdges"] = np.array(temporal_edges, dtype=object) if temporal_edges else np.empty((0, 3), dtype=object)
    G.graph["costmaps"]   = costmaps
    G.graph["instruction"] = instruction
    G.graph["cfg"] = {"episode_dir": str(episode_dir)}

    print(f"\n[Graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    out_path = episode_dir / out_name
    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[Saved] {out_path}")
    return G


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a LangGeoNet-enriched topometric graph for one episode."
    )
    parser.add_argument("episode_dir",    help="Path to the episode directory")
    parser.add_argument("checkpoint",     help="Path to LangGeoNet .pth checkpoint")
    parser.add_argument(
        "--out",
        default="nodes_langgeo_topometric.pickle",
        help="Output pickle filename (saved inside episode_dir)",
    )
    parser.add_argument(
        "--min-pixels", type=int, default=10,
        help="Minimum mask pixel count to include a node (default: 10)",
    )
    parser.add_argument(
        "--da-window", type=int, default=5,
        help="Max frame gap for data-association edges (default: 5)",
    )
    args = parser.parse_args()

    build_langgeo_topometric_graph(
        episode_dir=args.episode_dir,
        checkpoint_path=args.checkpoint,
        out_name=args.out,
        min_mask_pixels=args.min_pixels,
        da_window=args.da_window,
    )
