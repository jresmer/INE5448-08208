import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def save_comparative_plot(data_dict, metrics_keys, title, ylabel, output_path, is_percentage=True):
    """Save a comparative bar plot for the requested metrics.

    :param data_dict: Mapping of configuration names to metric values.
    :type data_dict: dict
    :param metrics_keys: Metric names to plot.
    :type metrics_keys: list[str]
    :param title: Plot title.
    :type title: str
    :param ylabel: Y-axis label.
    :type ylabel: str
    :param output_path: Destination path for the saved plot.
    :type output_path: Path
    :param is_percentage: Whether the metrics are percentage-like values.
    :type is_percentage: bool
    """
    configs = list(data_dict.keys())
    if not configs:
        return

    x = np.arange(len(metrics_keys))
    width = 0.8 / len(configs)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, config in enumerate(configs):
        values = [data_dict[config].get(m, 0.0) for m in metrics_keys]
        offset = (i - len(configs)/2) * width + width/2
        ax.bar(x + offset, values, width, label=config)
        
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys, rotation=15, ha="right")
    
    if is_percentage:
        ax.set_ylim(0, 1.1)
        
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def extract_metrics(entries):
    """Aggregate evaluation metrics from a list of report entries.

    :param entries: Report entries to summarize.
    :type entries: list[dict]
    :returns: A dictionary with aggregated metrics.
    :rtype: dict
    """
    total = len(entries)
    if total == 0:
        return {}

    timeouts = sum(1 for e in entries if "ERROR: Evaluation timed out" in e.get("response", ""))
    valid_entries = [e for e in entries if "ERROR: Evaluation timed out" not in e.get("response", "")]
    valid_count = len(valid_entries)

    timeout_rate = timeouts / total if total > 0 else 0.0

    if valid_count == 0:
        return {
            "CRR": 0.0,
            "ASR": 0.0,
            "Timeout Rate": timeout_rate,
            "Safe Refusal": 0.0,
            "Citation Precision": 0.0,
            "Groundedness": 0.0,
            "Latency (s)": 0.0,
            "Char Count": 0.0
        }

    crr = sum(e.get("response_status", 0) for e in valid_entries) / valid_count
    asr = sum(e.get("attack_status", 0) for e in valid_entries) / valid_count

    safe_refusals = sum(1 for e in valid_entries if e.get("metrics", {}).get("refusal_triggered", False))
    safe_refusal_rate = safe_refusals / valid_count

    verified_citations = sum(1 for e in valid_entries if e.get("metrics", {}).get("citations_verified", False))
    citation_precision = verified_citations / valid_count

    groundedness = sum(e.get("metrics", {}).get("groundedness_similarity", 0.0) for e in valid_entries) / valid_count
    latency = sum(e.get("metrics", {}).get("latency_seconds", 0.0) for e in valid_entries) / valid_count
    char_count = sum(e.get("metrics", {}).get("char_count", 0) for e in valid_entries) / valid_count

    return {
        "CRR": crr,
        "ASR": asr,
        "Timeout Rate": timeout_rate,
        "Safe Refusal": safe_refusal_rate,
        "Citation Precision": citation_precision,
        "Groundedness": groundedness,
        "Latency (s)": latency,
        "Char Count": char_count
    }

def run_report_analysis():
    """Generate comparative plots from the available JSON report files."""
    reports_dir = Path("reports")
    if not reports_dir.exists() or not reports_dir.is_dir():
        reports_dir = Path("report")
        if not reports_dir.exists() or not reports_dir.is_dir():
            print("[ERROR] Report directory not found.")
            sys.exit(1)

    graphs_dir = reports_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    target_files = [f for f in reports_dir.glob("*.json") if f.name not in ["casos_de_teste.json", "documentos_envenenados.json"]]
    
    if not target_files:
        print("[ERROR] No valid report JSON files found to compare.")
        sys.exit(1)

    overall_data_by_config = {}
    category_data_by_config = defaultdict(dict)

    for file_path in target_files:
        config_name = file_path.stem
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        if not isinstance(data, list):
            continue

        overall_data_by_config[config_name] = extract_metrics(data)

        categories = {entry.get("category", entry.get("categoria", "unknown")) for entry in data}
        for cat in categories:
            cat_entries = [e for e in data if e.get("category", e.get("categoria")) == cat]
            category_data_by_config[cat][config_name] = extract_metrics(cat_entries)

    percentage_metrics = ["CRR", "ASR", "Timeout Rate", "Safe Refusal", "Citation Precision", "Groundedness"]
    absolute_metrics = ["Latency (s)", "Char Count"]

    save_comparative_plot(
        overall_data_by_config, 
        percentage_metrics, 
        "Overall Comparative Metrics (Percentages)", 
        "Percentage / Ratio", 
        graphs_dir / "comparative_overall_percentages.png", 
        is_percentage=True
    )

    save_comparative_plot(
        overall_data_by_config, 
        absolute_metrics, 
        "Overall Comparative Metrics (Absolutes)", 
        "Value", 
        graphs_dir / "comparative_overall_absolutes.png", 
        is_percentage=False
    )

    for cat, config_metrics in category_data_by_config.items():
        save_comparative_plot(
            config_metrics, 
            percentage_metrics, 
            f"Comparative Metrics - Category: {cat} (Percentages)", 
            "Percentage / Ratio", 
            graphs_dir / f"comparative_category_{cat}_percentages.png", 
            is_percentage=True
        )

        save_comparative_plot(
            config_metrics, 
            absolute_metrics, 
            f"Comparative Metrics - Category: {cat} (Absolutes)", 
            "Value", 
            graphs_dir / f"comparative_category_{cat}_absolutes.png", 
            is_percentage=False
        )

    print(f"[OK] Comparative analysis complete. Graphs saved to: {graphs_dir.resolve()}")

if __name__ == "__main__":
    run_report_analysis()