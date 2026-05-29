"""
Retarget episodes defined in episode_stats.csv to robot motion.

Each row in the CSV maps (pt_file, track_id, start_frame, end_frame) to an episode.
This script extracts the relevant frames per episode, runs SMPL body model forward pass,
and retargets to the target robot using GMR.

Usage:
    python scripts/episode_retarget_from_csv.py \
        --csv /path/to/episode_stats.csv \
        --output_dir ALIA_test_smplx \
        --robot unitree_g1 \
        --n_episodes 5
"""

import argparse
import csv
import os
import pathlib
import pickle
import sys

import numpy as np
import smplx
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import get_comotion_data_offline_fast
from rich import print

SMPLX_FOLDER = str(pathlib.Path(__file__).parent.parent / "assets" / "body_models")


def extract_episode(data: dict, track_id: int, start_frame: int, end_frame: int):
    """Return pose/trans/betas tensors for a single episode slice."""
    mask = (
        (data["id"] == track_id)
        & (data["frame_idx"] >= start_frame)
        & (data["frame_idx"] <= end_frame)
    )
    if mask.sum() == 0:
        return None
    pose = data["pose"][mask].numpy()          # (N, 72)
    trans = data["trans"][mask].numpy()        # (N, 3)
    betas = data["betas"][mask].numpy()        # (N, 10)
    return pose, trans, betas


def build_smplx_output(pose, trans, betas_per_frame, src_fps):
    betas_mean = betas_per_frame.mean(axis=0)  # (10,)
    global_orient = pose[:, :3]                # (N, 3)
    body_pose = pose[:, 3:66]                  # (N, 63)

    body_model = smplx.create(
        SMPLX_FOLDER, "smplx", gender="neutral", use_pca=False, num_betas=len(betas_mean)
    )

    num_frames = pose.shape[0]
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
        "pose_body": body_pose,
        "betas": betas_mean,
        "root_orient": global_orient,
        "trans": trans,
        "mocap_frame_rate": torch.tensor(src_fps),
    }
    human_height = 1.66 + 0.1 * float(betas_mean[0])
    return smplx_data, body_model, smplx_output, human_height


def retarget_episode(row: dict, pt_cache: dict, robot: str, output_dir: str,
                     src_fps: int, tgt_fps: int) -> None:
    pt_file = row["file"]
    track_id = int(row["track_id"])
    episode_id = row["episode_id"]
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    cam = row["cam"]
    filename_stem = os.path.splitext(row["filename"])[0]

    print(f"[bold cyan]Episode:[/bold cyan] {episode_id} | {cam} | track={track_id} "
          f"frames={start_frame}-{end_frame}")

    if pt_file not in pt_cache:
        print(f"  Loading {pt_file} ...")
        pt_cache[pt_file] = torch.load(pt_file, map_location="cpu", weights_only=False)
    data = pt_cache[pt_file]

    result = extract_episode(data, track_id, start_frame, end_frame)
    if result is None:
        print(f"  [red]No frames found — skipping.[/red]")
        return
    pose, trans, betas_per_frame = result
    print(f"  Extracted {len(pose)} frames")

    smplx_data, body_model, smplx_output, human_height = build_smplx_output(
        pose, trans, betas_per_frame, src_fps
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

    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
    dof_pos = np.array([q[7:] for q in qpos_list])

    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
        "track_id": track_id,
        "episode_id": episode_id,
        "cam": cam,
        "source_file": pt_file,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    out_name = f"{episode_id}.pkl"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"  [green]Saved:[/green] {out_path}  ({len(qpos_list)} frames @ {aligned_fps:.1f} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to episode_stats.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save retargeted .pkl files")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Number of CSV rows to process (default: all)")
    parser.add_argument("--src_fps", type=int, default=25)
    parser.add_argument("--tgt_fps", type=int, default=25)
    args = parser.parse_args()

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.n_episodes is not None and i >= args.n_episodes:
                break
            rows.append(row)

    print(f"Processing {len(rows)} episodes -> {args.output_dir}")

    pt_cache: dict = {}
    for row in rows:
        retarget_episode(row, pt_cache, args.robot, args.output_dir, args.src_fps, args.tgt_fps)

    print(f"\n[bold green]Done.[/bold green] {len(rows)} episodes processed.")


if __name__ == "__main__":
    main()
