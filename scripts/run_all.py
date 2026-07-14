#!/usr/bin/env python
"""一键运行全部 5 个数据集的训练 + 评估，汇总结果。

用法:
    python scripts/run_all.py                         # 全部训练+评估
    python scripts/run_all.py --skip_train            # 仅评估已有检查点
    python scripts/run_all.py --skip_eval             # 仅训练
    python scripts/run_all.py --datasets solar,taxi   # 只跑指定数据集
"""

import argparse
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Using `json`-module")

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 全部 5 个数据集配置
ALL_DATASETS = [
    ("solar",       "configs/solar.yaml"),
    ("electricity", "configs/electricity.yaml"),
    ("traffic",     "configs/traffic.yaml"),
    ("taxi",        "configs/taxi.yaml"),
    ("wikipedia",   "configs/wikipedia.yaml"),
]

# 论文表 1 参考值 (CRPS-sum, MSE)
PAPER_REF = {
    "solar":       (0.253, 7.7e2),
    "electricity": (0.021, 1.6e5),
    "traffic":     (0.040, 4.1e-4),
    "taxi":        (0.125, 2.2),
    "wikipedia":   (0.061, 2.92e7),
}


def run_cmd(cmd: str, desc: str) -> int:
    """运行命令并实时输出，同时将输出写入日志文件。

    使用 Popen 逐行读取 stdout，确保子进程输出通过本进程的
    print() 函数，从而被 TeeLogger 捕获到日志文件中。
    """
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f">>> {cmd}")
    print(f"{'='*60}")
    t0 = time.time()

    # 使用 Popen 逐行读取，使输出经过本进程的 print → 被 Tee 捕获
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        # 去掉行尾换行符，print 会自己加
        print(line.rstrip("\n").rstrip("\r"))

    process.wait()
    elapsed = time.time() - t0
    if process.returncode != 0:
        print(f"[!] 失败 (exit={process.returncode})，耗时 {elapsed:.0f}s")
    else:
        print(f"[OK] 完成，耗时 {elapsed:.0f}s")
    return process.returncode


def parse_eval_output(output: str) -> dict:
    """从 evaluate.py 输出中提取 CRPS-sum 和 MSE。"""
    result = {"crps_sum": None, "mse": None}
    for line in output.split("\n"):
        if "CRPS-sum:" in line:
            parts = line.split()
            try:
                result["crps_sum"] = float(parts[1])
            except (IndexError, ValueError):
                pass
        if "MSE:" in line and "CRPS" not in line:
            parts = line.split()
            try:
                result["mse"] = float(parts[1])
            except (IndexError, ValueError):
                pass
    return result


def main():
    parser = argparse.ArgumentParser(description="一键运行 LDT 全流程")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评估")
    parser.add_argument("--datasets", type=str, default=None,
                        help="指定数据集，逗号分隔 (如 solar,electricity)")
    parser.add_argument("--device", type=str, default=None, help="设备 cuda/cpu")
    args = parser.parse_args()

    # 确定要跑的数据集
    if args.datasets:
        names = [n.strip() for n in args.datasets.split(",")]
        datasets = [(n, f"configs/{n}.yaml") for n in names]
    else:
        datasets = ALL_DATASETS

    device_flag = f" --device {args.device}" if args.device else ""

    # 启动日志记录
    dataset_list = args.datasets.replace(",", "_") if args.datasets else "all"
    log_path = setup_logger(prefix=f"run_all_{dataset_list}")
    print(f"日志文件: {log_path}\n")

    results = {}

    for name, config_path in datasets:
        print(f"\n{'#'*60}")
        print(f"### 数据集: {name.upper()} ###")
        print(f"{'#'*60}")

        stage1_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", f"{name}_stage1", "best_model.pt")
        stage2_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", f"{name}_stage2", "best_model.pt")

        # ============================================================
        # 训练
        # ============================================================
        if not args.skip_train:
            # 第一阶段
            if os.path.exists(stage1_ckpt):
                print(f"[*] {name}: 第一阶段检查点已存在，跳过")
            else:
                ret = run_cmd(
                    f'python scripts/train.py --config {config_path} --stage 1{device_flag}',
                    f"{name} - 第一阶段 VAE 训练",
                )
                if ret != 0:
                    print(f"[!] {name}: 第一阶段失败，跳过后续")
                    continue

            # 第二阶段
            if os.path.exists(stage2_ckpt):
                print(f"[*] {name}: 第二阶段检查点已存在，跳过")
            else:
                ret = run_cmd(
                    f'python scripts/train.py --config {config_path} --stage 2{device_flag}',
                    f"{name} - 第二阶段 LDT 训练",
                )
                if ret != 0:
                    print(f"[!] {name}: 第二阶段失败，跳过评估")
                    continue

        # ============================================================
        # 评估
        # ============================================================
        if not args.skip_eval:
            if not os.path.exists(stage1_ckpt) or not os.path.exists(stage2_ckpt):
                print(f"[!] {name}: 检查点不完整，跳过评估")
                continue

            ret = run_cmd(
                f'python scripts/evaluate.py --config {config_path} '
                f'--stage1_ckpt {stage1_ckpt} --stage2_ckpt {stage2_ckpt} '
                f'--num_samples 100{device_flag}',
                f"{name} - 测试集评估",
            )

            # 单独运行评估以捕获输出（用于解析指标）
            eval_cmd = (
                f'python scripts/evaluate.py --config {config_path} '
                f'--stage1_ckpt {stage1_ckpt} --stage2_ckpt {stage2_ckpt} '
                f'--num_samples 100 --max_batches 200{device_flag}'
            )
            eval_result = subprocess.run(
                eval_cmd, shell=True, cwd=PROJECT_ROOT,
                capture_output=True, text=True,
            )
            # 将子进程输出也写入终端（进而写入日志）
            combined = eval_result.stdout + eval_result.stderr
            if combined.strip():
                print(f"\n[评估 {name} 子进程输出]")
                print(combined)
            parsed = parse_eval_output(combined)
            results[name] = parsed

    # ============================================================
    # 汇总结果
    # ============================================================
    if results:
        print("\n")
        print("=" * 90)
        print("最终结果汇总 (对比论文表 1)")
        print("=" * 90)
        print(f"{'数据集':<15} {'CRPS-sum (ours)':<18} {'CRPS-sum (paper)':<18} {'MSE (ours)':<16} {'MSE (paper)':<16}")
        print("-" * 90)

        for name, _ in datasets:
            r = results.get(name)
            ref = PAPER_REF.get(name, (None, None))
            if r and r["crps_sum"] is not None:
                cs_str = f"{r['crps_sum']:.4f}"
                mse_str = f"{r['mse']:.4e}" if r["mse"] else "N/A"
                ref_cs = f"{ref[0]}" if ref[0] else "-"
                ref_mse = f"{ref[1]}" if ref[1] else "-"
                print(f"{name:<15} {cs_str:<18} {ref_cs:<18} {mse_str:<16} {ref_mse:<16}")
            else:
                print(f"{name:<15} {'N/A':<18} {'-':<18} {'N/A':<16} {'-':<16}")

        print("-" * 90)
        print("paper 参考值来自论文表 1 (AAAI 2024)")
    else:
        print("\n[!] 没有评估结果可汇总")


if __name__ == "__main__":
    main()
