"""
Retarget SMPL tracking .pt files (from input_data/) to robot motion.

Each .pt file contains SMPL pose estimates with keys:
  id, pose (N,72), trans (N,3), betas (N,10), frame_idx

Usage (single file):
    python scripts/smpl_tracking_to_robot.py \
        --pt_file input_data/Vid1/<name>.pt \
        --robot unitree_g1 \
        --save_path output/Vid1/<name>.pkl

Usage (batch — all .pt files under input_data/):
    python scripts/smpl_tracking_to_robot.py \
        --batch \
        --input_dir input_data \
        --output_dir output \
        --robot unitree_g1
"""

import argparse
import os
import pathlib
import pickle
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    load_smpl_tracking_file,
    get_gvhmr_data_offline_fast,
)
from rich import print

HERE = pathlib.Path(__file__).parent
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"


def retarget_file(pt_file: str, robot: str, save_path: str, src_fps: int = 25,
                  tgt_fps: int = 30, visualize: bool = False) -> None:
    print(f"[bold]Processing:[/bold] {pt_file}")

    smplx_data, body_model, smplx_output, human_height = load_smpl_tracking_file(
        pt_file, str(SMPLX_FOLDER), src_fps=src_fps
    )
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=robot,
    )

    if visualize:
        from general_motion_retargeting import RobotMotionViewer
        viewer = RobotMotionViewer(
            robot_type=robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
        )

    qpos_list = []
    for i, smplx_frame in enumerate(smplx_data_frames):
        qpos = retarget.retarget(smplx_frame)
        qpos_list.append(qpos)
        if visualize:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                show_human_body_name=False,
            )

    if visualize:
        viewer.close()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
    }
    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"[green]Saved:[/green] {save_path}  ({len(qpos_list)} frames @ {aligned_fps:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(description="Retarget SMPL tracking .pt files to robot motion.")
    parser.add_argument("--pt_file", type=str, default=None, help="Single .pt file to process.")
    parser.add_argument("--batch", action="store_true", help="Process all .pt files under --input_dir.")
    parser.add_argument("--input_dir", type=str, default="input_data",
                        help="Root directory containing Vid*/  subdirectories with .pt files.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output root directory (mirrors input_dir structure).")
    parser.add_argument("--robot", default="unitree_g1",
                        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "booster_k1",
                                 "stanford_toddy", "fourier_n1", "engineai_pm01", "kuavo_s45",
                                 "hightorque_hi", "galaxea_r1pro", "unitree_h1", "unitree_h1_2"])
    parser.add_argument("--src_fps", type=int, default=25, help="Source video fps (default: 25).")
    parser.add_argument("--tgt_fps", type=int, default=30, help="Target retargeting fps (default: 30).")
    parser.add_argument("--visualize", action="store_true", help="Show MuJoCo viewer during retargeting.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Output .pkl path (single-file mode only).")
    args = parser.parse_args()

    if args.batch:
        input_root = pathlib.Path(args.input_dir)
        output_root = pathlib.Path(args.output_dir)
        pt_files = sorted(input_root.rglob("*.pt"))
        if not pt_files:
            print(f"[red]No .pt files found under {input_root}[/red]")
            return
        print(f"Found {len(pt_files)} .pt file(s) to process.")
        for pt_file in pt_files:
            rel = pt_file.relative_to(input_root)
            save_path = str(output_root / rel.with_suffix("").with_name(rel.stem + f"_{args.robot}.pkl"))
            retarget_file(str(pt_file), args.robot, save_path, args.src_fps, args.tgt_fps, args.visualize)
    elif args.pt_file:
        save_path = args.save_path or args.pt_file.replace(".pt", f"_{args.robot}.pkl")
        retarget_file(args.pt_file, args.robot, save_path, args.src_fps, args.tgt_fps, args.visualize)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
