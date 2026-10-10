"""Generate a PGFPlots scientific figure from verified saved cost records.

Requires only standard-library Python and the existing local TeX packages.
Run audit_section_5_4.py first; compile the emitted standalone TeX twice.
Historical results and figures are not modified.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACE = HERE / "SECTION_5_4_CLAIMS.json"


def main():
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    raw_path = "results/expAC10/raw_results.json"
    if hashlib.sha256((ROOT / raw_path).read_bytes()).hexdigest() != trace["source_sha256"][raw_path]:
        raise ValueError("Cost raw data changed after verification")
    configs = trace["cost_configurations"]
    by_id = {r["id"]: r for r in configs}
    groups = [
        ("direct_implicit_midpoint", None, "violet", "square", "Direct midpoint"),
        ("adaptive_implicit_midpoint", None, "orange!85!black", "triangle*", "Adaptive midpoint"),
        ("manual_sundman", None, "green!55!black", "*", "Manual $g=r$"),
        ("learned_monitor", "raw", "red!75!black", "x", "Unfactored raw"),
        ("learned_monitor", "invariant", "blue!70!black", "diamond", "Unfactored invariant"),
    ]
    lines = [r"\documentclass[border=4pt]{standalone}", r"\usepackage{pgfplots}",
             r"\usepgfplotslibrary{groupplots}", r"\usetikzlibrary{calc}",
             r"\pgfplotsset{compat=1.18}", r"\begin{document}", r"\begin{tikzpicture}",
             r"\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.5cm,group name=costgrid},",
             r"width=8cm,height=6.1cm,xmode=log,ymode=log,grid=major,grid style={gray!20},",
             r"xlabel={Field-call count},label style={font=\small},tick label style={font=\footnotesize},",
             r"title style={font=\small},xmin=950,xmax=25000,legend columns=3,",
             r"legend style={draw=none,font=\footnotesize,column sep=0.35cm}]"]
    for panel, (metric, title, ylabel, front_key) in enumerate([
        ("energy", "(a) Energy and work", "Geometric mean maximum energy error", "energy_vs_evaluations"),
        ("closure", "(b) State closure and work", "Geometric mean closure error", "closure_vs_evaluations"),
    ]):
        options = f"title={{{title}}},ylabel={{{ylabel}}}"
        if panel == 0:
            options += ",legend to name=costlegend"
        lines.append(r"\nextgroupplot[" + options + "]")
        for method, architecture, color, mark, label in groups:
            selected = [r for r in configs if r["method"] == method and
                        (architecture is None or r["architecture"] == architecture)]
            # Raw 'x' explicitly has a visible stroke; no facecolor masking.
            coords = " ".join(f"({r['field']:.13g},{r[metric]:.13g})" for r in selected)
            lines.append(r"\addplot+[only marks,mark=" + mark + f",color={color},mark size=2.4pt,"
                         + r"mark options={solid,line width=0.8pt}] coordinates {" + coords + "};")
            if panel == 0:
                lines.append(r"\addlegendentry{" + label + "}")
        front = [by_id[k] for k in trace["cost_fronts_original_counts"][front_key]]
        coords = " ".join(f"({r['field']:.13g},{r[metric]:.13g})" for r in front)
        lines.append(r"\addplot+[black,densely dotted,mark=none,line width=0.9pt] coordinates {" + coords + "};")
        if panel == 0:
            lines.append(r"\addlegendentry{18/18-complete front}")
    lines.extend([r"\end{groupplot}",
                  r"\node[anchor=north] at ($(costgrid c1r1.south)!0.5!(costgrid c2r1.south)+(0,-1.15cm)$) {\pgfplotslegendfromname{costlegend}};",
                  r"\end{tikzpicture}", r"\end{document}"])
    output = HERE / "fig_accuracy_cost_new.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"generator": "new/make_section_5_4_figure.py", "data": raw_path,
                "data_sha256": trace["source_sha256"][raw_path],
                "configuration_ids": list(by_id), "fronts": trace["cost_fronts_original_counts"],
                "scope": "Original counts and completed-run geometric means; incomplete groups are excluded from fronts.",
                "raw_marker_correction": "Visible stroked x markers for all 15 raw configurations.",
                "tex_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (HERE / "FIGURE_5_4_SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
