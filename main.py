#!/usr/bin/env python3
"""
Building Simulation Model - Main Entry Point

This is a thin wrapper that provides backward compatibility with the original
CLI interface while delegating to the refactored bsm package.

Usage:
    python main.py [config_file]

    If no config_file is provided, uses 'simulation_config.json' by default.

Example:
    python main.py simulation_config.json
    python main.py bestest_configs/case_600_config.json
"""

import argparse
import sys

from bsm.runner import run_simulation_from_config


def main():
    """Main entry point for the building simulation."""
    parser = argparse.ArgumentParser(
        description="Run a single-zone thermal simulation from a JSON config file."
    )
    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        default='simulation_config.json',
        help="Path to the simulation JSON configuration file (default: simulation_config.json)"
    )

    args = parser.parse_args()
    run_simulation_from_config(args.config_file)


if __name__ == '__main__':
    main()
