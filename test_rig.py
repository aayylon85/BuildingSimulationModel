#!/usr/bin/env python3
"""
Building Simulation Test Rig - Entry Point

This is a thin wrapper that provides backward compatibility with the original
CLI interface while delegating to the refactored bsm package.

Usage:
    # Run full BESTEST suite
    python test_rig.py --bestest

    # Run specific cases
    python test_rig.py --bestest --cases 600 610 620

    # Custom output directory
    python test_rig.py --bestest --output results/my_test_run
"""

import argparse
import sys

# Import the test infrastructure from the bsm package
from bsm.test_rig import BESTESTRunner


def main():
    """Main entry point for the test rig."""
    parser = argparse.ArgumentParser(
        description="Building Simulation Test Rig - Run BESTEST validation suite"
    )
    parser.add_argument(
        '--bestest',
        action='store_true',
        help='Run ASHRAE 140 BESTEST validation suite'
    )
    parser.add_argument(
        '--cases',
        nargs='+',
        help='Specific BESTEST case IDs to run (e.g., 600 620 900 920)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/bestest_suite',
        help='Output directory for results (default: results/bestest_suite)'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default=None,
        help='Path to reference results for comparison'
    )

    args = parser.parse_args()

    if not args.bestest:
        parser.print_help()
        print("\nError: Please specify --bestest to run BESTEST validation suite")
        sys.exit(1)

    # Run BESTEST
    runner = BESTESTRunner(
        output_base_dir=args.output,
        reference_results_path=args.reference
    )

    results = runner.run_full_bestest(case_ids=args.cases)

    # Exit with error code if any tests failed
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
