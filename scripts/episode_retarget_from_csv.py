"""
Retarget episodes defined in episode_stats.csv to robot motion.

Each row in the CSV maps (pt_file, track_id, start_frame, end_frame) to an episode.
This script extracts the relevant frames per episode, runs SMPL body model forward pass,
and retargets to the target robot using GMR.

Usage:
    python scripts/episode_retarget_from_csv.py \
        --csv /path/to/episode_stats.csv \
        --output_dir outputs/pkls \
        --robot unitree_g1 \
        --n_episodes 5

Parallel processing (default: all CPU cores):
    python scripts/episode_retarget_from_csv.py \
        --csv /path/to/episode_stats.csv \
        --output_dir outputs/pkls \
        --n_workers 96
"""

import argparse
import csv
import multiprocessing
import os
import pathlib
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import smplx
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import get_comotion_data_offline_fast

DEFAULT_SMPLX_FOLDER = str(pathlib.Path(__file__).parent.parent / "assets" / "body_models")


def extract_episode(data: dict, track_id: int, start_frame: int, end_frame: int):
    """Return pose/trans/betas tensors for a single episode slice."""
    mask = (
        (data["id"] == track_id)
        & (data["frame_idx"] >= start_frame)
        & (data["frame_idx"] <= end_frame)
    )
    if mask.sum() == 0:
        return None
    pose  = data["pose"][mask].numpy()   # (N, 72)
    trans = data["trans"][mask].numpy()  # (N, 3)
    betas = data["betas"][mask].numpy()  # (N, 10)
    return pose, trans, betas


def build_smplx_output(pose, trans, betas_per_frame, src_fps, smplx_folder=DEFAULT_SMPLX_FOLDER):
    betas_mean    = betas_per_frame.mean(axis=0)  # (10,)
    global_orient = pose[:, :3]                   # (N, 3)
    body_pose     = pose[:, 3:66]                 # (N, 63)

    if not os.path.isdir(smplx_folder):
        raise FileNotFoundError(
            f"SMPL-X body models folder not found: {smplx_folder}\n"
            f"  Copy assets/body_models/ from the source machine, or pass --smplx_folder "
            f"pointing to a directory that contains an 'smplx/' subfolder."
        )

    body_model = smplx.create(
        smplx_folder, "smplx", gender="neutral", use_pca=False, num_betas=len(betas_mean)
    )

    num_frames  = pose.shape[0]
    smplx_output = body_model(
        betas=torch.tensor(betas_mean).float().view(1, -1),
        global_orient=torch.tensor(global_orient).float(),
        body_pose=torch.tensor(body_pose).float(),
        transl=torch.tensor(trans).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )

    smplx_data = {
        "pose_body":         body_pose,
        "betas":             betas_mean,
        "root_orient":       global_orient,
        "trans":             trans,
        "mocap_frame_rate":  torch.tensor(src_fps),
    }
    human_height = 1.66 + 0.1 * float(betas_mean[0])
    return smplx_data, body_model, smplx_output, human_height


def retarget_episode(row: dict, pt_cache: dict, robot: str, output_dir: str,
                     src_fps: int, tgt_fps: int,
                     smplx_folder: str = DEFAULT_SMPLX_FOLDER) -> bool:
    """
    Retarget one episode. Returns True on success, False if no frames found.
    Raises on hard errors.
    """
    pt_file     = row["file"]
    track_id    = int(row["track_id"])
    episode_id  = row["episode_id"]
    start_frame = int(row["start_frame"])
    end_frame   = int(row["end_frame"])

    if pt_file not in pt_cache:
        pt_cache[pt_file] = torch.load(pt_file, map_location="cpu", weights_only=False)
    data = pt_cache[pt_file]

    result = extract_episode(data, track_id, start_frame, end_frame)
    if result is None:
        return False
    pose, trans, betas_per_frame = result

    smplx_data, body_model, smplx_output, human_height = build_smplx_output(
        pose, trans, betas_per_frame, src_fps, smplx_folder=smplx_folder
    )
    smplx_frames, aligned_fps = get_comotion_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=robot,
    )

    qpos_list = [retarget.retarget(frame, offset_to_ground=True) for frame in smplx_frames]

    root_pos = np.array([q[:3]                    for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]]     for q in qpos_list])  # wxyz → xyzw
    dof_pos  = np.array([q[7:]                     for q in qpos_list])

    motion_data = {
        "fps":          aligned_fps,
        "root_pos":     root_pos,
        "root_rot":     root_rot,
        "dof_pos":      dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
        "track_id":     track_id,
        "episode_id":   episode_id,
        "cam":          row.get("cam", ""),
        "source_file":  pt_file,
        "start_frame":  start_frame,
        "end_frame":    end_frame,
    }

    out_path = os.path.join(output_dir, f"{episode_id}.pkl")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(motion_data, f)
    return True


# ── Multiprocessing worker ────────────────────────────────────────────────────

def _init_worker():
    """Pin each worker to a single thread so numpy/OpenBLAS doesn't over-subscribe cores."""
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _make_chunks(rows: list, chunk_size: int) -> list[list]:
    """
    Group rows by .pt file, then split each group into chunks of chunk_size.
    Rows within a chunk share the same .pt file → one load covers all of them.
    """
    groups: defaultdict[str, list] = defaultdict(list)
    for row in rows:
        groups[row["file"]].append(row)
    chunks = []
    for grp in groups.values():
        for i in range(0, len(grp), chunk_size):
            chunks.append(grp[i : i + chunk_size])
    return chunks


def _worker(args: tuple) -> tuple[int, int, int]:
    """
    Process one chunk of episodes (all from the same .pt file).
    Returns (n_done, n_skipped, n_failed).
    """
    rows, robot, output_dir, src_fps, tgt_fps, smplx_folder = args
    pt_cache: dict = {}
    n_done = n_skipped = n_failed = 0

    for row in rows:
        out_path = os.path.join(output_dir, f"{row['episode_id']}.pkl")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        try:
            ok = retarget_episode(
                row, pt_cache, robot, output_dir, src_fps, tgt_fps, smplx_folder
            )
            if ok:
                n_done += 1
            else:
                n_failed += 1
                print(f"[WARN] No frames: episode {row['episode_id']}", flush=True)
        except Exception as exc:
            n_failed += 1
            print(f"[ERROR] episode {row['episode_id']}: {exc}", flush=True)

    return n_done, n_skipped, n_failed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retarget episodes from CSV to robot motion .pkl files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",         required=True, help="Path to episode_stats.csv")
    parser.add_argument("--output_dir",  required=True, help="Directory to save .pkl files")
    parser.add_argument("--robot",       default="unitree_g1")
    parser.add_argument("--src_fps",     type=int, default=25)
    parser.add_argument("--tgt_fps",     type=int, default=25)
    parser.add_argument("--n_episodes",  type=int, default=None,
                        help="Limit number of CSV rows to process (default: all)")
    parser.add_argument("--n_workers",   type=int, default=None,
                        help="Parallel workers (default: all CPU cores)")
    parser.add_argument("--smplx_folder", default=DEFAULT_SMPLX_FOLDER,
                        help="Path to body_models dir containing an 'smplx/' subfolder")
    args = parser.parse_args()

    # ── Load rows ─────────────────────────────────────────────────────────────
    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.n_episodes is not None and i >= args.n_episodes:
                break
            rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)

    n_workers = args.n_workers or multiprocessing.cpu_count()

    # Auto-tune chunk size: aim for ~4× more work items than workers so all
    # cores stay busy and load balances well.  Each chunk shares one .pt load.
    chunk_size = max(1, len(rows) // (n_workers * 4))
    chunks     = _make_chunks(rows, chunk_size)
    n_workers  = min(n_workers, len(chunks))

    n_unique_pt = len({r["file"] for r in rows})
    print(f"Episodes  : {len(rows)}")
    print(f"PT files  : {n_unique_pt}")
    print(f"Chunks    : {len(chunks)}  (chunk_size={chunk_size})")
    print(f"Workers   : {n_workers}")
    print(f"Output    : {args.output_dir}")

    work_items = [
        (chunk, args.robot, args.output_dir, args.src_fps, args.tgt_fps, args.smplx_folder)
        for chunk in chunks
    ]

    # ── Dispatch ──────────────────────────────────────────────────────────────
    total_done = total_skipped = total_failed = 0

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        futures = [executor.submit(_worker, item) for item in work_items]
        with tqdm(total=len(rows), unit="ep", desc="retarget") as pbar:
            for future in as_completed(futures):
                try:
                    n_done, n_skipped, n_failed = future.result()
                except Exception as exc:
                    print(f"[ERROR] Worker crashed: {exc}", flush=True)
                    n_done = n_skipped = 0
                    n_failed = 1
                total_done    += n_done
                total_skipped += n_skipped
                total_failed  += n_failed
                pbar.update(n_done + n_skipped + n_failed)

    print(f"\nDone — processed: {total_done}  skipped: {total_skipped}  failed: {total_failed}")


if __name__ == "__main__":
    main()
