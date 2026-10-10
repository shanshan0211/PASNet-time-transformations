"""Display all saved configuration-level RSS measurements; no new benchmark."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    audit = json.loads((HERE / "REVIEW_EVIDENCE_AUDIT.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise RuntimeError("saved evidence must pass audit first")
    path = ROOT / "results/expAC10/raw_results.json"
    expected = audit["source_sha256"]["results/expAC10/raw_results.json"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("cost records changed after audit")
    rows = audit["memory_configurations"]
    groups = [("direct_implicit_midpoint", None, "Direct midpoint", "#9467bd", "s"),
              ("adaptive_implicit_midpoint", None, "Adaptive midpoint", "#ff7f0e", "^"),
              ("manual_sundman", None, "Manual g=r", "#2ca02c", "o"),
              ("learned_monitor", "raw", "Learned raw", "#d62728", "x"),
              ("learned_monitor", "invariant", "Learned invariant", "#1f77b4", "D")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for method, arch, label, color, marker in groups:
        selected = [r for r in rows if r["method"] == method and (arch is None or r["architecture"] == arch)]
        ax.scatter([r["peak_rss_mib"] for r in selected], [r["complete"] / r["registered"] for r in selected],
                   c=color, marker=marker, s=38, linewidths=1, label=label, alpha=.85)
    ax.set(xlabel="Fresh-process configuration peak RSS (MiB)", ylabel="Completion fraction (of 18 scenarios)", ylim=(-.04,1.08))
    ax.grid(alpha=.2)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(.5,1.02))
    fig.tight_layout()
    outputs = []
    for extension in ("pdf", "png"):
        path = HERE / ("fig_configuration_memory." + extension)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    provenance = {"data": "results/expAC10/raw_results.json", "data_sha256": expected,
                  "generator": "new/make_memory_figure.py", "configurations": rows, "outputs": outputs,
                  "scope": "All 38 historical configuration processes; RSS is not per-rollout allocation or isolated model memory."}
    (HERE / "FIGURE_MEMORY_SOURCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Saved configuration-memory figure and provenance")


if __name__ == "__main__":
    main()
