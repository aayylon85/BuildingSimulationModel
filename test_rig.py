"""
Building Simulation Test Rig

Comprehensive testing framework for running batch simulations, generating datasets,
and validating against BESTEST standards.

Usage:
    # Run full BESTEST suite
    python test_rig.py --bestest

    # Run specific cases
    python test_rig.py --bestest --cases 600 610 620

    # Custom output directory
    python test_rig.py --bestest --output results/my_test_run
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import warnings
import glob

# Import existing simulation infrastructure
from main import run_simulation_from_config
from bestest.cases.case_library import BESTESTCaseLibrary
from bestest.utils.config_builder import BESTESTConfigBuilder


@dataclass
class TestConfiguration:
    """Represents a single simulation test configuration"""
    case_id: str
    config_path: str
    description: str
    output_name: Optional[str] = None
    reference_data: Optional[Dict] = None


@dataclass
class TestResult:
    """Results from a single simulation run"""
    case_id: str
    config_path: str
    success: bool
    error_message: Optional[str] = None
    results_csv_path: Optional[str] = None
    config_copy_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class TestRunner:
    """Runs simulation tests in batch mode"""

    def __init__(self, output_base_dir: str = "results"):
        self.output_base_dir = Path(output_base_dir)
        self.results_collector = ResultsCollector()

    def run_single_test(self, config: TestConfiguration) -> TestResult:
        """Execute a single simulation test"""
        try:
            print(f"  Running simulation for case {config.case_id}...")

            # Run the simulation
            run_simulation_from_config(config.config_path)

            # Find the results CSV file that was just created
            # The simulation saves results to results/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS_*days_results.csv
            results_dir = Path("results")

            # Get the most recent results directory
            date_dirs = [d for d in results_dir.glob("*") if d.is_dir()]
            if not date_dirs:
                raise RuntimeError("No results directory found after simulation")

            latest_dir = max(date_dirs, key=os.path.getmtime)

            # Find the most recent CSV file in that directory
            csv_files = list(latest_dir.glob("*_results.csv"))
            if not csv_files:
                raise RuntimeError(f"No results CSV found in {latest_dir}")

            results_csv_path = max(csv_files, key=os.path.getmtime)

            # Extract metrics from results
            df = self.results_collector.load_results(str(results_csv_path))
            metrics = self.results_collector.extract_metrics(df)

            print(f"   Success - Heating: {metrics['annual_heating_kwh']:.1f} kWh, "
                  f"Cooling: {metrics['annual_cooling_kwh']:.1f} kWh")

            return TestResult(
                case_id=config.case_id,
                config_path=config.config_path,
                success=True,
                results_csv_path=str(results_csv_path),
                metrics=metrics
            )

        except Exception as e:
            print(f"   Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return TestResult(
                case_id=config.case_id,
                config_path=config.config_path,
                success=False,
                error_message=str(e)
            )

    def run_batch_tests(self, configs: List[TestConfiguration]) -> List[TestResult]:
        """Execute multiple tests with progress tracking"""
        results = []

        print(f"\n{'='*70}")
        print(f"Running {len(configs)} test case(s)")
        print(f"{'='*70}\n")

        for i, config in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] Test Case: {config.case_id} - {config.description}")
            result = self.run_single_test(config)
            results.append(result)
            print()

        # Summary
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes

        print(f"{'='*70}")
        print(f"Batch Results: {successes} succeeded, {failures} failed")
        print(f"{'='*70}\n")

        return results


class ResultsCollector:
    """Extracts metrics and aggregates results"""

    @staticmethod
    def load_results(result_path: str) -> pd.DataFrame:
        """Load CSV results into DataFrame"""
        return pd.read_csv(result_path)

    @staticmethod
    def extract_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key performance metrics"""
        # Calculate timestep from time column
        dt_hours = df['Time (hrs)'].diff().mean()

        # Energy metrics (convert W to kWh)
        heating_power = df['HVAC Power (W)'].clip(lower=0)
        cooling_power = -df['HVAC Power (W)'].clip(upper=0)

        metrics = {
            'annual_heating_kwh': (heating_power * dt_hours).sum() / 1000,
            'annual_cooling_kwh': (cooling_power * dt_hours).sum() / 1000,
            'peak_heating_w': heating_power.max(),
            'peak_cooling_w': cooling_power.max(),
            'max_zone_temp_c': df['Zone Temp (C)'].max(),
            'min_zone_temp_c': df['Zone Temp (C)'].min(),
            'mean_zone_temp_c': df['Zone Temp (C)'].mean(),
        }
        return metrics

    def aggregate_results(self, results: List[TestResult]) -> pd.DataFrame:
        """Create summary table from multiple test results"""
        rows = []
        for result in results:
            if result.success and result.metrics:
                rows.append({
                    'Case ID': result.case_id,
                    'Heating (kWh)': result.metrics['annual_heating_kwh'],
                    'Cooling (kWh)': result.metrics['annual_cooling_kwh'],
                    'Peak Heating (W)': result.metrics['peak_heating_w'],
                    'Peak Cooling (W)': result.metrics['peak_cooling_w'],
                    'Max Temp (�C)': result.metrics['max_zone_temp_c'],
                    'Min Temp (�C)': result.metrics['min_zone_temp_c'],
                    'Mean Temp (�C)': result.metrics['mean_zone_temp_c'],
                })
            else:
                rows.append({
                    'Case ID': result.case_id,
                    'Heating (kWh)': 'FAILED',
                    'Cooling (kWh)': 'FAILED',
                    'Peak Heating (W)': 'FAILED',
                    'Peak Cooling (W)': 'FAILED',
                    'Max Temp (�C)': 'FAILED',
                    'Min Temp (�C)': 'FAILED',
                    'Mean Temp (�C)': 'FAILED',
                })
        return pd.DataFrame(rows)

    def save_summary(self, summary: pd.DataFrame, output_path: str):
        """Save aggregated results as CSV"""
        summary.to_csv(output_path, index=False)
        print(f"Summary saved to: {output_path}")


class BESTESTRunner(TestRunner):
    """Specialized runner for BESTEST validation suite"""

    # Cases currently supported by the simulator
    SUPPORTED_CASES = {'600', '620', '900', '920'}

    # Cases requiring features not yet implemented
    UNSUPPORTED_CASES = {
        '610': 'Requires shading geometry (overhang/fins) - approximated with reduced SHGC',
        '630': 'Requires shading geometry (overhang/fins) - approximated with reduced SHGC',
        '640': 'Requires setback schedules (night setback)',
        '650': 'Requires scheduled ventilation (night ventilation)',
        '910': 'Requires shading geometry (overhang/fins) - approximated with reduced SHGC',
        '930': 'Requires shading geometry (overhang/fins) - approximated with reduced SHGC',
        '940': 'Requires setback schedules (night setback)',
        '950': 'Requires scheduled ventilation (night ventilation)',
    }

    def __init__(self, output_base_dir: str = "results/bestest_suite",
                 reference_results_path: Optional[str] = None):
        super().__init__(output_base_dir)
        self.case_library = BESTESTCaseLibrary()
        self.config_builder = BESTESTConfigBuilder()
        self.reference_results = self._load_reference_results(reference_results_path)

    def _load_reference_results(self, path: Optional[str]) -> Optional[Dict]:
        """Load ASHRAE 140-2020 reference data if available"""
        if path and Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _discover_bestest_cases(self) -> List[str]:
        """Find all available BESTEST cases"""
        self.case_library.load_from_json('bestest/cases/section_5_2_cases.json')
        return list(self.case_library.cases.keys())

    def run_full_bestest(self, case_ids: Optional[List[str]] = None) -> List[TestResult]:
        """Run all or specified BESTEST cases"""
        # Discover available cases
        all_case_ids = self._discover_bestest_cases()

        if case_ids is None:
            case_ids = all_case_ids
        else:
            # Validate requested cases exist
            invalid = set(case_ids) - set(all_case_ids)
            if invalid:
                print(f"Warning: Unknown case IDs: {invalid}")
                case_ids = [c for c in case_ids if c in all_case_ids]

        if not case_ids:
            print("Error: No valid BESTEST cases to run")
            return []

        # Filter to only supported cases
        requested_unsupported = [c for c in case_ids if c in self.UNSUPPORTED_CASES]
        case_ids = [c for c in case_ids if c in self.SUPPORTED_CASES]

        print(f"\nDiscovered {len(all_case_ids)} BESTEST cases")
        print(f"Currently supported: {len(self.SUPPORTED_CASES)} cases ({', '.join(sorted(self.SUPPORTED_CASES))})")
        print(f"Currently unsupported: {len(self.UNSUPPORTED_CASES)} cases (features not yet implemented)")

        if requested_unsupported:
            print(f"\n{'='*70}")
            print("SKIPPING UNSUPPORTED CASES:")
            print(f"{'='*70}")
            for case_id in sorted(requested_unsupported):
                reason = self.UNSUPPORTED_CASES[case_id]
                print(f"  Case {case_id}: {reason}")
            print(f"{'='*70}\n")

        if not case_ids:
            print("Error: No supported BESTEST cases to run")
            print("Supported cases: " + ", ".join(sorted(self.SUPPORTED_CASES)))
            return []

        print(f"Running {len(case_ids)} supported case(s): {', '.join(sorted(case_ids))}\n")

        # Generate configurations for each case
        configs = []
        for case_id in case_ids:
            # Generate config for this case
            case_spec = self.case_library.get_case(case_id)
            config_dict = self.config_builder.build_config(case_spec)

            # Save temporary config file
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            config_path = self.output_base_dir / f"case_{case_id}_config.json"

            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            configs.append(TestConfiguration(
                case_id=case_id,
                config_path=str(config_path),
                description=case_spec.description
            ))

        # Run all tests
        results = self.run_batch_tests(configs)

        # Save summary
        self._save_bestest_summary(results)

        return results

    def _save_bestest_summary(self, results: List[TestResult]):
        """Save BESTEST results summary"""
        collector = ResultsCollector()
        summary_df = collector.aggregate_results(results)

        # Save CSV
        summary_path = self.output_base_dir / 'bestest_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nBESTEST summary saved to: {summary_path}")

        # Generate markdown report
        report = self.generate_validation_report(results)
        report_path = self.output_base_dir / 'bestest_validation_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Validation report saved to: {report_path}")

    def generate_validation_report(self, results: List[TestResult]) -> str:
        """Create markdown validation report"""
        collector = ResultsCollector()
        summary_df = collector.aggregate_results(results)

        report = "# BESTEST Validation Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Standard:** ASHRAE 140-2020 Section 5.2\n\n"

        report += f"## Summary\n\n"
        successes = sum(1 for r in results if r.success)
        report += f"- **Total cases run:** {len(results)}\n"
        report += f"- **Successful:** {successes}\n"
        report += f"- **Failed:** {len(results) - successes}\n\n"

        # Document simulator capabilities
        report += "## Simulator Capabilities\n\n"
        report += f"**Currently Supported Cases:** {', '.join(sorted(self.SUPPORTED_CASES))}\n\n"

        if self.UNSUPPORTED_CASES:
            report += "**Cases Not Yet Implemented** (features under development):\n\n"
            report += "| Case | Reason |\n"
            report += "|------|--------|\n"
            for case_id in sorted(self.UNSUPPORTED_CASES.keys()):
                reason = self.UNSUPPORTED_CASES[case_id]
                report += f"| {case_id} | {reason} |\n"
            report += "\n"

        if len(results) - successes > 0:
            report += "## Failed Cases\n\n"
            for r in results:
                if not r.success:
                    report += f"- **{r.case_id}:** {r.error_message}\n"
            report += "\n"

        report += "## Results Table\n\n"
        report += summary_df.to_markdown(index=False)
        report += "\n\n"

        report += "## Notes\n\n"
        report += "- Heating/Cooling energy in kWh over simulation period\n"
        report += "- Peak heating/cooling in Watts\n"
        report += "- Temperature statistics in �C\n"
        report += "- All case definitions are available in the test suite for future implementation\n"

        return report


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Building Simulation Test Rig',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full BESTEST suite
  python test_rig.py --bestest

  # Run specific cases
  python test_rig.py --bestest --cases 600 610 900

  # Custom output directory
  python test_rig.py --bestest --output results/my_test
        """
    )

    parser.add_argument('--bestest', action='store_true',
                       help='Run BESTEST validation suite')
    parser.add_argument('--cases', nargs='+',
                       help='Specific BESTEST case IDs to run (e.g., 600 610 900)')
    parser.add_argument('--output', default='results/bestest_suite',
                       help='Output directory (default: results/bestest_suite)')
    parser.add_argument('--reference',
                       help='Path to ASHRAE 140 reference results JSON (optional)')

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
