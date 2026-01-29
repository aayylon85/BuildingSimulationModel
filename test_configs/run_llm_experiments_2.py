#!/usr/bin/env python3
"""
Run multiple LLM agent simulations for winter and summer conditions.

Usage:
    python test_configs/run_llm_experiments.py
"""

import subprocess
import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration
EXPERIMENTS = [
    # (config_file, num_runs, description)
    ("test_configs/llm_agent_test_config_winter.json", 3, "Winter (Jan 07)")
]


def run_simulation(config_path: str, run_number: int, description: str) -> bool:
    """Run a single simulation and return success status."""
    print(f"\n{'='*60}")
    print(f"Starting {description} - Run {run_number}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, "main.py", config_path],
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode == 0:
            print(f"\n[OK] {description} Run {run_number} completed successfully")
            return True
        else:
            print(f"\n[FAILED] {description} Run {run_number} failed with code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n[ERROR] {description} Run {run_number} error: {e}")
        return False


def main():
    """Run all experiments."""
    print("="*60)
    print("LLM Agent Experiment Runner")
    print("="*60)

    results = []

    for config_file, num_runs, description in EXPERIMENTS:
        for run_num in range(1, num_runs + 1):
            success = run_simulation(config_file, run_num, description)
            results.append((description, run_num, success))

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    successful = sum(1 for _, _, s in results if s)
    total = len(results)

    for description, run_num, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {description} - Run {run_num}")

    print(f"\nTotal: {successful}/{total} simulations completed successfully")

    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
