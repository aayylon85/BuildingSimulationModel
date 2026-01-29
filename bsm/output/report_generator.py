"""
Report generator for building simulation results.

Generates comprehensive markdown reports with energy summaries,
temperature statistics, and embedded figures.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class SimulationReportGenerator:
    """Generate comprehensive simulation reports."""

    def __init__(
        self,
        json_summary_path: str,
        figures_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the report generator.

        Args:
            json_summary_path: Path to the JSON summary file
            figures_dir: Directory containing saved figures
            config_path: Path to the configuration file (for additional details)
        """
        self.json_summary_path = json_summary_path
        self.figures_dir = figures_dir
        self.config_path = config_path

        # Load the JSON summary
        with open(json_summary_path, 'r') as f:
            self.summary = json.load(f)

        # Load config if provided
        self.config = None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)

    def generate_markdown_report(self) -> str:
        """
        Generate a comprehensive markdown report.

        Returns:
            Markdown report as a string
        """
        lines = []

        # Header
        lines.append("# Simulation Report")
        lines.append("")

        # Simulation info
        sim_info = self.summary.get("simulation_info", {})
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Simulation Start:** {sim_info.get('start_datetime', 'N/A')}")
        lines.append(f"**Duration:** {sim_info.get('duration_days', 'N/A')} days")
        lines.append(f"**Timestep:** {sim_info.get('timestep_minutes', 'N/A')} minutes")
        lines.append(f"**Config File:** `{sim_info.get('config_file', 'N/A')}`")
        lines.append(f"**LLM Agents:** {'Enabled' if sim_info.get('llm_agents_enabled') else 'Disabled'}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Energy Summary
        lines.append("## Energy Summary")
        lines.append("")
        energy = self.summary.get("energy_summary_kwh", {})
        peak = self.summary.get("peak_power_w", {})

        lines.append("| Category | Total Energy (kWh) | Peak Power (W) |")
        lines.append("|----------|-------------------|----------------|")
        lines.append(f"| Heating | {energy.get('heating', 0):.2f} | {peak.get('heating', 0):.0f} |")
        lines.append(f"| Cooling | {energy.get('cooling', 0):.2f} | {peak.get('cooling', 0):.0f} |")
        lines.append(f"| **Total HVAC** | **{energy.get('total_hvac', 0):.2f}** | - |")
        # Always show equipment and lighting (for all simulation modes)
        lines.append(f"| Equipment | {energy.get('equipment', 0):.2f} | {peak.get('equipment', 0):.0f} |")
        lines.append(f"| Lighting | {energy.get('lighting', 0):.2f} | {peak.get('lighting', 0):.0f} |")
        lines.append(f"| **Total** | **{energy.get('total', 0):.2f}** | **{peak.get('total', 0):.0f}** |")
        lines.append("")

        # Temperature Performance
        lines.append("## Temperature Performance")
        lines.append("")
        temp = self.summary.get("temperature_stats_c", {})
        lines.append(f"- **Mean Indoor Temperature:** {temp.get('mean', 'N/A')}°C")
        lines.append(f"- **Temperature Range:** {temp.get('min', 'N/A')}°C to {temp.get('max', 'N/A')}°C")
        lines.append(f"- **Comfort Hours:** {temp.get('comfort_hours_pct', 'N/A')}%")
        lines.append("")

        # Agent Activity (if LLM mode)
        if sim_info.get('llm_agents_enabled') and "agent_summary" in self.summary:
            lines.append("## Agent Activity")
            lines.append("")
            agent = self.summary.get("agent_summary", {})
            lines.append(f"- **Number of Agents:** {agent.get('num_agents', 'N/A')}")
            lines.append(f"- **Total Decision Rounds:** {agent.get('total_decisions', 'N/A')}")
            lines.append(f"- **Thermostat Adjustments:** {agent.get('thermostat_adjustments', 'N/A')}")
            lines.append(f"- **Window State Changes:** {agent.get('window_changes', 'N/A')}")
            lines.append("")

        # Figures section
        if self.figures_dir and os.path.exists(self.figures_dir):
            lines.append("## Figures")
            lines.append("")

            figure_files = [
                ("temperatures.png", "Temperature Profile"),
                ("energy_flows.png", "Energy Flows"),
                ("agent_activity.png", "Agent Activity"),
                ("summary_dashboard.png", "Summary Dashboard"),
            ]

            for filename, title in figure_files:
                fig_path = os.path.join(self.figures_dir, filename)
                if os.path.exists(fig_path):
                    # Use relative path for markdown
                    rel_path = f"figures/{filename}"
                    lines.append(f"### {title}")
                    lines.append(f"![{title}]({rel_path})")
                    lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by BuildingSimulationModel*")

        return "\n".join(lines)

    def save_report(self, output_path: str) -> str:
        """
        Generate and save the markdown report.

        Args:
            output_path: Path to save the report

        Returns:
            Path to the saved report
        """
        report = self.generate_markdown_report()

        with open(output_path, 'w') as f:
            f.write(report)

        return output_path


def generate_report_from_results(
    results_dir: str,
    base_filename: str,
) -> Optional[str]:
    """
    Convenience function to generate a report from simulation results.

    Args:
        results_dir: Directory containing simulation results
        base_filename: Base filename for the results (without extension)

    Returns:
        Path to the generated report, or None if generation failed
    """
    json_path = os.path.join(results_dir, f"{base_filename}_summary.json")
    config_path = os.path.join(results_dir, f"{base_filename}_config.json")
    figures_dir = os.path.join(results_dir, "figures")
    report_path = os.path.join(results_dir, f"{base_filename}_report.md")

    if not os.path.exists(json_path):
        print(f"Error: JSON summary not found at {json_path}")
        return None

    try:
        generator = SimulationReportGenerator(
            json_summary_path=json_path,
            figures_dir=figures_dir if os.path.exists(figures_dir) else None,
            config_path=config_path if os.path.exists(config_path) else None,
        )

        generator.save_report(report_path)
        print(f"Saved report to: {report_path}")
        return report_path

    except Exception as e:
        print(f"Error generating report: {e}")
        return None
