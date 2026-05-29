"""
Run the full CoMotion → robot retargeting + visualization pipeline.

Steps
-----
1. retarget   — CoMotion .pt files → per-episode robot motion .pkl
2. render     — .pkl → MuJoCo simulation video
3. clip       — source video → frame-range clip with bounding-box overlay
4. compare    — source clip + robot video → side-by-side comparison video

Quick start
-----------
    cp configs/pipeline.yaml.example configs/pipeline.yaml
    # edit configs/pipeline.yaml with paths for your machine
    python scripts/run_pipeline.py --config configs/pipeline.yaml

Override any YAML value on the command line:
    python scripts/run_pipeline.py --config configs/pipeline.yaml --n_episodes 5 --run_id quick_test

Skip individual steps (e.g. if you only want to re-render):
    python scripts/run_pipeline.py --config configs/pipeline.yaml --skip retarget clip
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

import yaml

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent


def run(cmd: list[str], step_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    print("  " + " ".join(cmd))
    print()
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  Completed in {elapsed:.1f}s")


def resolve(path: str) -> str:
    """Return path as-is if absolute, else relative to project root."""
    p = pathlib.Path(path)
    if p.is_absolute():
        return str(p)
    return str(ROOT / p)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full CoMotion→robot pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/pipeline.yaml",
                        help="YAML config file (default: configs/pipeline.yaml)")
    # Overrides — all optional; take precedence over YAML values
    parser.add_argument("--episode_csv")
    parser.add_argument("--video_mapping_csv")
    parser.add_argument("--robot")
    parser.add_argument("--n_episodes", type=int)
    parser.add_argument("--src_fps", type=int)
    parser.add_argument("--tgt_fps", type=int)
    parser.add_argument("--output_root")
    parser.add_argument("--run_id")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["retarget", "render", "clip", "compare"],
                        help="Steps to skip")
    parser.add_argument("--only", nargs="*",
                        choices=["retarget", "render", "clip", "compare"],
                        help="Run only these steps (overrides --skip)")
    args = parser.parse_args()

    # ── Load YAML config ─────────────────────────────────────────────────────
    config_path = resolve(args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        print(f"  Copy configs/pipeline.yaml.example to configs/pipeline.yaml and fill in paths.")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    for key in ["episode_csv", "video_mapping_csv", "robot", "n_episodes",
                "src_fps", "tgt_fps", "output_root", "run_id"]:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val

    # ── Resolve output directories ────────────────────────────────────────────
    run_root   = pathlib.Path(resolve(cfg["output_root"])) / cfg["run_id"]
    pkl_dir    = run_root / "pkls"
    robot_dir  = run_root / "videos" / "robot"
    source_dir = run_root / "videos" / "source"
    cmp_dir    = run_root / "videos" / "comparison"

    python = sys.executable

    # ── Determine which steps to run ─────────────────────────────────────────
    # Priority: --only > --skip > config steps > all steps
    all_steps = ["retarget", "render", "clip", "compare"]
    if args.only:
        active = set(args.only)
    elif args.skip:
        active = set(all_steps) - set(args.skip)
    elif cfg.get("steps"):
        active = set(cfg["steps"])
    else:
        active = set(all_steps)

    n_episodes = cfg.get("n_episodes")  # None means all rows
    print(f"Pipeline config  : {config_path}")
    print(f"Run root         : {run_root}")
    print(f"Episodes         : {'all' if n_episodes is None else n_episodes}")
    print(f"Robot            : {cfg['robot']}")
    print(f"Active steps     : {', '.join(s for s in all_steps if s in active)}")

    def n_ep_args():
        """Return --n_episodes flag only when a limit is set."""
        return ["--n_episodes", str(n_episodes)] if n_episodes is not None else []

    # ── Step 1: Retarget ──────────────────────────────────────────────────────
    if "retarget" in active:
        run([
            python, str(HERE / "episode_retarget_from_csv.py"),
            "--csv",        cfg["episode_csv"],
            "--output_dir", str(pkl_dir),
            "--robot",      cfg["robot"],
            "--src_fps",    str(cfg["src_fps"]),
            "--tgt_fps",    str(cfg["tgt_fps"]),
            *n_ep_args(),
        ], "retarget: CoMotion → robot .pkl")

    # ── Step 2: Render robot videos ───────────────────────────────────────────
    if "render" in active:
        run([
            python, str(HERE / "record_episode_videos.py"),
            "--motion_dir", str(pkl_dir),
            "--video_dir",  str(robot_dir),
            "--robot",      cfg["robot"],
        ], "render: .pkl → MuJoCo simulation videos")

    # ── Step 3: Clip source videos with bounding boxes ───────────────────────
    if "clip" in active:
        run([
            python, str(HERE / "clip_source_with_bb.py"),
            "--csv",           cfg["episode_csv"],
            "--video_mapping", cfg["video_mapping_csv"],
            "--output_dir",    str(source_dir),
            *n_ep_args(),
        ], "clip: source video → frame clip + bounding box")

    # ── Step 4: Side-by-side comparison ──────────────────────────────────────
    if "compare" in active:
        run([
            python, str(HERE / "make_comparison_video.py"),
            "--source_dir", str(source_dir),
            "--robot_dir",  str(robot_dir),
            "--output_dir", str(cmp_dir),
        ], "compare: source + robot → side-by-side video")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"  Output root: {run_root}")
    print(f"    pkls/              {pkl_dir}")
    print(f"    videos/robot/      {robot_dir}")
    print(f"    videos/source/     {source_dir}")
    print(f"    videos/comparison/ {cmp_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
