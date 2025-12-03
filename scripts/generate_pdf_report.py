#!/usr/bin/env python3
"""
generate_pdf_report.py - Generate PDF evaluation reports from CSV results.

This script reads CSV results from the results/ directory and uses a
Markdown template to create a comprehensive evaluation report. It can
optionally convert the Markdown to PDF using pandoc.

Usage:
    python scripts/generate_pdf_report.py \
        --results-dir results/ \
        --model-name tinyGPT \
        --domain "General Purpose" \
        --output report.md

    python scripts/generate_pdf_report.py \
        --results-dir results/ \
        --model-name tinyGPT \
        --domain "Healthcare" \
        --output report.pdf \
        --pdf

Example:
    # Generate Markdown report
    python scripts/generate_pdf_report.py --results-dir results/capstone \
        --model-name tinyGPT --domain "Healthcare" --output evaluation_report.md

    # Generate PDF report (requires pandoc)
    python scripts/generate_pdf_report.py --results-dir results/capstone \
        --model-name tinyGPT --domain "Healthcare" --output evaluation_report.pdf --pdf
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "templates",
    "eval_report_template.md",
)


# ==============================================================================
# CSV Reading Functions
# ==============================================================================


def read_results_from_csv(
    results_dir: str,
    model_name: str,
) -> Dict[str, pd.DataFrame]:
    """
    Read all CSV result files for a model from a directory.

    Args:
        results_dir: Directory containing CSV result files
        model_name: Name of the model to filter files

    Returns:
        Dictionary mapping file names to DataFrames
    """
    results = {}
    pattern = os.path.join(results_dir, f"{model_name}*.csv")

    for csv_file in glob.glob(pattern):
        name = os.path.basename(csv_file).replace(".csv", "")
        try:
            results[name] = pd.read_csv(csv_file)
            print(f"   Loaded: {csv_file}")
        except Exception as e:
            print(f"   Warning: Failed to read {csv_file}: {e}")

    # Also try pattern without model name prefix
    if not results:
        pattern = os.path.join(results_dir, "*.csv")
        for csv_file in glob.glob(pattern):
            name = os.path.basename(csv_file).replace(".csv", "")
            try:
                results[name] = pd.read_csv(csv_file)
                print(f"   Loaded: {csv_file}")
            except Exception as e:
                print(f"   Warning: Failed to read {csv_file}: {e}")

    return results


def merge_results(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple result DataFrames into a single DataFrame.

    Args:
        results: Dictionary of DataFrames

    Returns:
        Merged DataFrame with all results
    """
    if not results:
        return pd.DataFrame(columns=["benchmark", "metric", "value"])

    all_rows = []
    for name, df in results.items():
        if "benchmark" in df.columns and "metric" in df.columns:
            all_rows.append(df)
        elif "metric" in df.columns and "value" in df.columns:
            # Add source column if missing benchmark
            df_copy = df.copy()
            if "benchmark" not in df_copy.columns:
                df_copy["benchmark"] = name
            all_rows.append(df_copy)
        else:
            # Try to reshape the DataFrame
            for col in df.columns:
                if col not in ["benchmark", "metric", "value", "category", "name"]:
                    for idx, val in df[col].items():
                        all_rows.append(
                            pd.DataFrame(
                                [{"benchmark": name, "metric": col, "value": val}]
                            )
                        )

    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame(columns=["benchmark", "metric", "value"])


# ==============================================================================
# Table Generation Functions
# ==============================================================================


def generate_benchmark_tables(df: pd.DataFrame) -> str:
    """
    Generate Markdown tables for benchmark results.

    Args:
        df: DataFrame with benchmark results

    Returns:
        Markdown formatted tables
    """
    if df.empty:
        return "No benchmark results available.\n"

    # Filter for benchmark-type results
    benchmark_df = df[
        df["benchmark"].isin(
            ["accuracy", "truthfulqa", "robustness", "llm_judge", "compliance"]
        )
        | df.get("category", pd.Series(["benchmark"] * len(df))).eq("benchmark")
    ]

    if benchmark_df.empty:
        benchmark_df = df

    lines = ["| Benchmark | Metric | Value |", "|-----------|--------|-------|"]

    for _, row in benchmark_df.iterrows():
        benchmark = row.get("benchmark", row.get("name", "unknown"))
        metric = row.get("metric", "score")
        value = row.get("value", 0)

        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        lines.append(f"| {benchmark} | {metric} | {value_str} |")

    return "\n".join(lines) + "\n"


def generate_safety_tables(df: pd.DataFrame) -> str:
    """
    Generate Markdown tables for safety test results.

    Args:
        df: DataFrame with safety test results

    Returns:
        Markdown formatted tables
    """
    if df.empty:
        return "No safety test results available.\n"

    # Filter for safety-type results
    safety_df = df[
        df["benchmark"].isin(
            ["toxigen", "truthfulqa", "prescription_avoidance", "professional_referral"]
        )
        | df.get("category", pd.Series([""] * len(df))).eq("safety")
    ]

    if safety_df.empty:
        return "No safety test results available.\n"

    lines = ["| Safety Test | Metric | Value |", "|-------------|--------|-------|"]

    for _, row in safety_df.iterrows():
        test_name = row.get("benchmark", row.get("name", "unknown"))
        metric = row.get("metric", "score")
        value = row.get("value", 0)

        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)

        lines.append(f"| {test_name} | {metric} | {value_str} |")

    return "\n".join(lines) + "\n"


def generate_performance_tables(df: pd.DataFrame) -> str:
    """
    Generate Markdown tables for performance metrics.

    Args:
        df: DataFrame with performance metrics

    Returns:
        Markdown formatted tables
    """
    if df.empty:
        return "No performance metrics available.\n"

    # Filter for performance-type results
    perf_df = df[
        df["benchmark"].isin(["performance", "latency", "throughput"])
        | df.get("category", pd.Series([""] * len(df))).eq("performance")
        | df["metric"].str.contains("latency|throughput|memory", case=False, na=False)
    ]

    if perf_df.empty:
        return "No performance metrics available.\n"

    lines = ["| Metric | Value |", "|--------|-------|"]

    for _, row in perf_df.iterrows():
        metric = row.get("metric", "unknown")
        value = row.get("value", 0)

        if isinstance(value, float):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)

        lines.append(f"| {metric} | {value_str} |")

    return "\n".join(lines) + "\n"


# ==============================================================================
# Summary Generation Functions
# ==============================================================================


def generate_executive_summary(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Generate executive summary based on results.

    Args:
        df: DataFrame with all results
        config: Configuration dictionary

    Returns:
        Executive summary text
    """
    if df.empty:
        return "No evaluation results available for summary.\n"

    summary_lines = [
        f"This report presents the evaluation results for **{config.get('model_name', 'the model')}** "
        f"in the **{config.get('domain', 'general')}** domain.\n",
        "\n**Key Findings:**\n",
    ]

    # Extract key metrics
    for _, row in df.iterrows():
        benchmark = row.get("benchmark", row.get("name", ""))
        metric = row.get("metric", "")
        value = row.get("value", 0)

        if isinstance(value, float) and metric:
            if "ratio" in metric.lower() or "score" in metric.lower():
                summary_lines.append(f"- **{benchmark}**: {metric} = {value:.2%}\n")

    return "".join(summary_lines)


def generate_conclusion(
    df: pd.DataFrame,
    config: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate conclusion section based on results and thresholds.

    Args:
        df: DataFrame with all results
        config: Configuration dictionary
        thresholds: Optional threshold dictionary

    Returns:
        Conclusion text
    """
    conclusion_lines = []

    # Determine overall status
    overall_pass = True
    threshold_results = []

    if thresholds:
        for metric_key, threshold in thresholds.items():
            # Try to find the metric in the DataFrame
            matching = df[df["metric"].str.contains(metric_key, case=False, na=False)]
            if not matching.empty:
                actual = matching.iloc[0]["value"]
                if isinstance(actual, (int, float)):
                    passed = actual >= threshold
                    if not passed:
                        overall_pass = False
                    threshold_results.append(
                        {"metric": metric_key, "threshold": threshold, "actual": actual, "passed": passed}
                    )

    status = "✅ PASS" if overall_pass else "❌ FAIL"
    conclusion_lines.append(f"**Overall Status: {status}**\n\n")

    if threshold_results:
        conclusion_lines.append("### Threshold Analysis\n\n")
        conclusion_lines.append("| Metric | Threshold | Actual | Status |\n")
        conclusion_lines.append("|--------|-----------|--------|--------|\n")
        for result in threshold_results:
            status_emoji = "✅" if result["passed"] else "❌"
            conclusion_lines.append(
                f"| {result['metric']} | {result['threshold']:.2%} | "
                f"{result['actual']:.2%} | {status_emoji} |\n"
            )
        conclusion_lines.append("\n")

    conclusion_lines.append("### Recommendations\n\n")
    conclusion_lines.append("1. Review any failing threshold metrics and investigate root causes\n")
    conclusion_lines.append("2. Consider additional domain-specific benchmarks for comprehensive coverage\n")
    conclusion_lines.append("3. Run regression analysis against previous model versions\n")
    conclusion_lines.append("4. Document any known limitations for production deployment\n")

    return "".join(conclusion_lines)


# ==============================================================================
# Report Generation Functions
# ==============================================================================


def generate_markdown_report(
    results: Dict[str, pd.DataFrame],
    template_path: str,
    config: Dict[str, Any],
) -> str:
    """
    Generate Markdown report from template and results.

    Args:
        results: Dictionary of result DataFrames
        template_path: Path to Markdown template
        config: Configuration dictionary

    Returns:
        Generated Markdown report content
    """
    # Merge all results
    merged_df = merge_results(results)

    # Load template
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            template = f.read()
    else:
        # Use inline template if file not found
        template = """# LLM Evaluation Report: {model_name}

## Domain: {domain}

**Generated:** {timestamp}
**Evaluator:** BenchRight v1.0

---

## Executive Summary

{executive_summary}

---

## 1. Introduction

### 1.1 Evaluation Objectives

{objectives}

### 1.2 Model Under Test

| Property | Value |
|----------|-------|
| Model Name | {model_name} |
| Model Path | {model_path} |
| Domain | {domain} |

---

## 2. Benchmark Results

{benchmark_tables}

---

## 3. Safety Findings

{safety_tables}

---

## 4. Performance Metrics

{performance_tables}

---

## 5. Conclusion

{conclusion}

---

## Appendix

{appendix}

---

*Report generated by BenchRight LLM Evaluation Framework*
"""

    # Generate content sections
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    executive_summary = generate_executive_summary(merged_df, config)
    benchmark_tables = generate_benchmark_tables(merged_df)
    safety_tables = generate_safety_tables(merged_df)
    performance_tables = generate_performance_tables(merged_df)
    conclusion = generate_conclusion(merged_df, config, config.get("thresholds"))

    # Fill template
    report = template.format(
        model_name=config.get("model_name", "Unknown Model"),
        domain=config.get("domain", "General"),
        timestamp=timestamp,
        executive_summary=executive_summary,
        objectives=config.get("objectives", "Evaluate model for production readiness."),
        model_path=config.get("model_path", "N/A"),
        benchmark_tables=benchmark_tables,
        safety_tables=safety_tables,
        performance_tables=performance_tables,
        conclusion=conclusion,
        appendix=config.get("appendix", "No additional data."),
    )

    return report


def convert_to_pdf(markdown_path: str, pdf_path: str) -> bool:
    """
    Convert Markdown to PDF using pandoc.

    Args:
        markdown_path: Path to Markdown file
        pdf_path: Path for output PDF file

    Returns:
        True if conversion succeeded, False otherwise

    Note:
        Requires pandoc to be installed. If pandoc is not available,
        this function will print installation instructions.
    """
    # Try different PDF engines in order of preference
    pdf_engines = ["xelatex", "pdflatex", "wkhtmltopdf", "weasyprint"]

    for engine in pdf_engines:
        try:
            cmd = ["pandoc", markdown_path, "-o", pdf_path]
            # All engines need the --pdf-engine argument
            cmd.extend(["--pdf-engine", engine])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"   PDF generated using {engine}")
                return True
            else:
                continue

        except FileNotFoundError:
            pass

    # If we get here, pandoc is not available or all engines failed
    print("TODO: PDF generation requires pandoc to be installed.")
    print("Install pandoc with:")
    print("  Ubuntu/Debian: sudo apt-get install pandoc texlive-xetex")
    print("  macOS: brew install pandoc basictex")
    print("  Windows: choco install pandoc miktex")
    print("")
    print(f"Markdown report saved to: {markdown_path}")
    print("You can manually convert to PDF using:")
    print(f"  pandoc {markdown_path} -o {pdf_path}")

    return False


def save_report(
    content: str,
    output_path: str,
    generate_pdf: bool = False,
) -> None:
    """
    Save report to file(s).

    Args:
        content: Report content
        output_path: Output file path
        generate_pdf: Whether to also generate PDF
    """
    # Determine output paths
    if output_path.endswith(".pdf"):
        md_path = output_path.replace(".pdf", ".md")
        pdf_path = output_path
    elif output_path.endswith(".md"):
        md_path = output_path
        pdf_path = output_path.replace(".md", ".pdf")
    else:
        md_path = output_path + ".md"
        pdf_path = output_path + ".pdf"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or ".", exist_ok=True)

    # Save Markdown
    with open(md_path, "w") as f:
        f.write(content)
    print(f"   Markdown saved: {md_path}")

    # Generate PDF if requested
    if generate_pdf:
        convert_to_pdf(md_path, pdf_path)


# ==============================================================================
# CLI Interface
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation reports from CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate Markdown report:
    python scripts/generate_pdf_report.py --results-dir results/ \\
        --model-name tinyGPT --domain "Healthcare" --output report.md

  Generate PDF report (requires pandoc):
    python scripts/generate_pdf_report.py --results-dir results/ \\
        --model-name tinyGPT --domain "Healthcare" --output report.pdf --pdf

Note:
  PDF generation requires pandoc to be installed. If pandoc is not available,
  only the Markdown report will be generated.
        """,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing CSV result files",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model (used to filter CSV files and in report)",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="General Purpose",
        help="Evaluation domain (default: General Purpose)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (e.g., report.md or report.pdf)",
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate PDF in addition to Markdown (requires pandoc)",
    )

    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_TEMPLATE_PATH,
        help="Path to Markdown template file",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="N/A",
        help="Path to model file (for documentation)",
    )

    parser.add_argument(
        "--objectives",
        type=str,
        default="Evaluate model for production readiness.",
        help="Evaluation objectives (for report)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("BenchRight Report Generator")
    print("=" * 60)

    # Validate inputs
    if not os.path.isdir(args.results_dir):
        print(f"❌ Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"   Results Dir: {args.results_dir}")
    print(f"   Model Name:  {args.model_name}")
    print(f"   Domain:      {args.domain}")
    print(f"   Output:      {args.output}")
    print(f"   Generate PDF: {args.pdf}")

    # Read results
    print(f"\n📂 Reading results...")
    results = read_results_from_csv(args.results_dir, args.model_name)

    if not results:
        print(f"⚠️ Warning: No CSV files found for model '{args.model_name}'")
        print(f"   Searched in: {args.results_dir}")

    # Build configuration
    config = {
        "model_name": args.model_name,
        "domain": args.domain,
        "model_path": args.model_path,
        "objectives": args.objectives,
        "appendix": "See CSV files for raw data.",
    }

    # Generate report
    print(f"\n📝 Generating report...")
    report_content = generate_markdown_report(
        results=results,
        template_path=args.template,
        config=config,
    )

    # Save report
    print(f"\n💾 Saving report...")
    save_report(
        content=report_content,
        output_path=args.output,
        generate_pdf=args.pdf,
    )

    print(f"\n" + "=" * 60)
    print("✅ Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
