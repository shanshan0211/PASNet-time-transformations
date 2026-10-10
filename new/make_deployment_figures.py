"""Build publication figures from audited saved records, without dynamics.

Uses standard-library Python and PGFPlots. Historical artifacts are read-only.
Run audit_section_5_2.py and audit_section_5_3.py before this generator.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FAMILY = ("semi_major_axis", "eccentricity", "orientation_radians",
          "momentum_direction")
ARMS = ("invariant_e099", "sundman_e09", "sundman_e099")
COLORS = {"raw": "red!75!black", "invariant": "blue!70!black",
          "manual_r": "orange!85!black", "invariant_e099": "cyan!55!black",
          "sundman_e09": "green!50!black", "sundman_e099": "violet"}
LABELS = {"raw": "Raw, train $e=0.9$", "invariant": "Invariant, train $e=0.9$",
          "manual_r": "Manual $g=r$", "invariant_e099": "Unfactored, train $e=0.99$",
          "sundman_e09": "Factored, train $e=0.9$",
          "sundman_e099": "Factored, train $e=0.99$"}
MARKS = {"invariant_e099": "triangle", "sundman_e09": "*",
         "sundman_e099": "diamond", "manual_r": "square"}


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def verified_sources(trace_path):
    trace = read(trace_path)
    # Manuscript and audit-script hashes are not the data used to draw figures.
    hashes = {p: h for p, h in trace["source_sha256"].items()
              if p.startswith("results/")}
    for p, h in hashes.items():
        if digest(p) != h:
            raise ValueError(f"Source changed after audit: {p}")
    return hashes


def coords(points):
    return " ".join(f"({float(x):.12g},{float(y):.12g})" for x, y in points)


def plot(points, options):
    return r"\addplot[" + options + "] coordinates {" + coords(points) + "};"


def legend(label):
    return r"\addlegendentry{" + label + "}"


def start():
    return [
        r"\documentclass[border=5pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.65cm,vertical sep=2.6cm},",
        r"width=8.2cm,height=6.3cm,grid=major,grid style={gray!20},",
        r"label style={font=\small},tick label style={font=\footnotesize},",
        r"title style={font=\small},legend style={font=\scriptsize,draw=none,fill=white},",
        r"scaled ticks=false]",
    ]


def finish(lines, stem, manifest):
    lines += [r"\end{groupplot}\end{tikzpicture}\end{document}"]
    tex = HERE / (stem + ".tex")
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest["generator"] = "new/make_deployment_figures.py"
    manifest["tex_sha256"] = hashlib.sha256(tex.read_bytes()).hexdigest()
    (HERE / (stem + "_SOURCE.json")).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(tex)


def mechanism():
    hashes = verified_sources("new/SECTION_5_2_CLAIMS.json")
    raw = read("results/expAC19/raw_results.json")
    baseline = read("results/expAC9/raw_results.json")["records"]
    lines = start()
    lines.append(r"\nextgroupplot[title={(a) Saved monitor profiles},xmode=log,ymode=log,"
                 r"xlabel={Reference radius $r$},ylabel={Monitor $g$},xmin=0.009,xmax=2.1,"
                 r"legend pos=south east]")
    lines.append(r"\draw[fill=gray!12,draw=none] (rel axis cs:0,0) rectangle"
                 r" (axis cs:0.1,\pgfkeysvalueof{/pgfplots/ymax});")
    counts = Counter()
    for architecture in ("raw", "invariant", "manual_r"):
        selected = [r for r in raw["orbit_records"]
                    if r["architecture"] == architecture
                    and r["semi_major_axis"] == 1 and r["eccentricity"] == .99
                    and r["orientation_radians"] == 0
                    and r["momentum_direction"] == "forward"]
        assert len(selected) == (1 if architecture == "manual_r" else 5)
        for i, r in enumerate(selected):
            profile = r["profile"]
            for leg in (0, 1):
                points = sorted((radius, g) for anomaly, radius, g in
                                zip(profile["eccentric_anomaly"], profile["radius"],
                                    profile["monitor"])
                                if (anomaly < math.pi) == (leg == 0))
                lines.append(plot(points, "mark=none,thin,color=" + COLORS[architecture]
                                  + ("" if i == 0 and leg == 0 else ",forget plot")))
                if i == 0 and leg == 0:
                    lines.append(legend({"raw": "Raw (5 checkpoints)",
                                         "invariant": "Invariant (5 checkpoints)",
                                         "manual_r": "Manual $g=r$"}[architecture]))
            counts[architecture] += 1
    lines.append(r"\draw[densely dashed] (axis cs:0.1,\pgfkeysvalueof{/pgfplots/ymin})"
                 r" -- (axis cs:0.1,\pgfkeysvalueof{/pgfplots/ymax});")
    lines.append(r"\node[anchor=north west,font=\scriptsize,fill=white] at (rel axis cs:0.02,0.98)"
                 r" {$a=1,\ e=0.99$; training $r_{\min}=0.1$};")

    lines.append(r"\nextgroupplot[title={(b) Retrospective reference diagnostic},"
                 r"ymode=log,xmin=-0.6,xmax=8.6,ymax=45,"
                 r"xtick={0,1,2,3,4,5,6,7,8},"
                 r"xticklabels={R,I,M,R,I,M,R,I,M},"
                 r"xlabel={\shortstack{R: raw; I: invariant; M: manual\\Groups: $e=0.7,\ 0.9,\ 0.99$}},"
                 r"ylabel={$\rho_{\max}^{\mathrm{ref}}$ at $h=0.04$},"
                 r"legend pos=north west]")
    outcomes = defaultdict(list)
    for r in baseline:
        k = (r["architecture"], r["checkpoint_seed"], *(r[f] for f in FAMILY))
        outcomes[k].append(r["completion_status"] == "complete")
    groups = defaultdict(list)
    family_counts = Counter()
    for r in raw["orbit_records"]:
        k = (r["architecture"], r["checkpoint_seed"], *(r[f] for f in FAMILY))
        status = outcomes[k]
        assert len(status) == 2
        tag = "complete" if all(status) else "failed" if not any(status) else "mixed"
        family_counts[tag] += 1
        pos = 3 * (.7, .9, .99).index(r["eccentricity"]) + (
            "raw", "invariant", "manual_r").index(r["architecture"])
        # Deterministic horizontal offsets display families; they carry no metric.
        offset = (((len(groups[(r["architecture"], tag)]) * 17) % 61) / 60 - .5) * .42
        groups[(r["architecture"], tag)].append((pos + offset,
                                                r["spectral_radius_maximum"]))
    for (arch, tag), points in groups.items():
        mark = {"complete": "o", "failed": "x", "mixed": "diamond*"}[tag]
        lines.append(plot(points, f"only marks,mark={mark},mark size=1.6pt,"
                          f"color={COLORS[arch]},mark options={{line width=0.6pt}}"
                          + ",forget plot"))
    for mark, label in (("o", "Both phases complete"), ("x", "Both phases fail"),
                        ("diamond*", "One phase fails")):
        lines.append(r"\addlegendimage{only marks,mark=" + mark + ",black}")
        lines.append(legend(label))
    lines.append(plot([(-.6, 1), (8.6, 1)], "black,densely dashed,mark=none,forget plot"))

    ratios = sorted(r["failure_radius_over_periapsis"] for r in raw["failure_locations"])
    assert len(ratios) == 252
    lines.append(r"\nextgroupplot[title={(c) Reference projections of failures},"
                 r"xmode=log,xmin=0.95,xmax=9,ymin=0,ymax=1.03,"
                 r"xtick={1,2,5,8},xticklabels={1,2,5,8},xlabel={$r_{\mathrm{ref}}(t_{\mathrm{fail}})/r_p$},"
                 r"ylabel={Cumulative fraction of 252 failures}]")
    lines.append(plot([(ratios[0], 0)] + [(x, (i + 1) / len(ratios))
                                        for i, x in enumerate(ratios)],
                      "const plot,mark=none,color=violet,thick,forget plot"))
    for x in (2, 5):
        n = sum(y <= x for y in ratios)
        lines.append(plot([(x, 0), (x, n / 252)], "gray,densely dashed,mark=none,forget plot"))
        lines.append(r"\node[anchor=south east,font=\scriptsize] at (axis cs:"
                     + f"{x},{n / 252:.12g})" + " {" + f"{n}/252" + "};")

    lines.append(r"\nextgroupplot[title={(d) Resolution-scale calculation},"
                 r"xmode=log,ymode=log,x dir=reverse,xmin=0.009,xmax=0.11,"
                 r"xtick={0.01,0.1},xticklabels={0.01,0.1},"
                 r"xlabel={Radius $r$},ylabel={$\chi(r)/\chi(0.1)$},"
                 r"legend pos=north west]")
    radii = [.1 * 10 ** (-i / 80) for i in range(81)]
    lines.append(plot([(r, (.1 / r) ** 1.5) for r in radii],
                      "mark=none,thick,color=red!75!black"))
    lines.append(legend("Constant monitor"))
    lines.append(plot([(r, (.1 / r) ** .5) for r in radii],
                      "mark=none,thick,color=orange!85!black"))
    lines.append(legend("Radial monitor $g=r$"))
    lines.append(r"\node[anchor=north east,font=\scriptsize] at (axis cs:0.011,31.62) {31.6};")
    lines.append(r"\node[anchor=south east,font=\scriptsize] at (axis cs:0.011,3.162) {3.16};")
    finish(lines, "fig_periapsis_new", {
        "source_sha256": hashes, "profiles": dict(counts),
        "family_outcomes": dict(family_counts), "reference_families": 396,
        "rollout_labels": 792, "failure_projections": 252,
        "scope": "Panels a-c use saved records; d is a normalized analytic scale calculation, not a spectral-radius fit.",
        "selection": "Profiles: a=1, e=.99, orientation=0, forward momentum; both radial legs retained.",
        "diagnostic_unit": "One point per checkpoint-orbit family; paired initial-anomaly outcomes use separate mixed markers.",
    })


def structure():
    hashes = verified_sources("new/SECTION_5_3_CLAIMS.json")
    baseline = read("results/expAC9/raw_results.json")["records"]
    records = {a: [r for r in baseline if r["architecture"] == a]
               for a in ("raw", "invariant", "manual_r")}
    diagnostics = {a: [] for a in ARMS}
    for arm in ARMS:
        records[arm] = []
        for directory in sorted((ROOT / "results/expAC20/parts").glob(arm + "_seed*")):
            records[arm].extend(json.loads((directory / "rollouts.json").read_text())["records"])
            diagnostics[arm].extend(json.loads((directory / "diagnostic.json").read_text())["orbit_records"])
        assert len(records[arm]) == 360 and len(diagnostics[arm]) == 180
    lines = start()
    names = [LABELS[a] for a in ("raw", "invariant", *ARMS, "manual_r")]
    lines.append(r"\nextgroupplot[title={(a) Completion by deployment eccentricity},"
                 r"title style={yshift=0.6cm},"
                 r"xbar,xmin=0,xmax=1.06,ymin=-0.6,ymax=5.6,"
                 r"ytick={0,1,2,3,4,5},yticklabels={"
                 + ",".join("{" + n + "}" for n in names) + r"},"
                 r"yticklabel style={font=\scriptsize,text width=2.6cm,align=right},"
                 r"xlabel={Completion fraction},bar width=4pt,"
                 r"legend style={at={(0.5,1.02)},anchor=south,legend columns=3,font=\scriptsize}]")
    completions = {}
    for i, e in enumerate((.7, .9, .99)):
        points = []
        for j, arm in enumerate(("raw", "invariant", *ARMS, "manual_r")):
            chosen = [r for r in records[arm] if r["eccentricity"] == e]
            n = sum(r["completion_status"] == "complete" for r in chosen)
            completions[f"{arm}|{e}"] = {"complete": n, "n": len(chosen)}
            points.append((n / len(chosen), j))
        lines.append(plot(points, "xbar,area legend,bar shift=" + str((i - 1) * 4.5)
                          + "pt,draw=none,fill=" + ("gray!45", "blue!65", "red!65")[i]))
        lines.append(legend("$e=" + str(e) + "$"))

    lines.append(r"\nextgroupplot[title={(b) Diagnostics before new deployments},"
                 r"ymode=log,xmin=-0.5,xmax=8.5,xtick={0,1,2,3,4,5,6,7,8},"
                 r"xticklabels={.7,.9,.99,.7,.9,.99,.7,.9,.99},"
                 r"xlabel={Deployment $e$: unfactored / factored .9 / factored .99},"
                 r"xlabel style={font=\scriptsize},ylabel={$\rho_{\max}^{\mathrm{ref}}$ at $h=0.04$},"
                 r"title style={yshift=1.05cm},"
                 r"legend style={at={(0.5,1.02)},anchor=south}]")
    family_outcomes = Counter()
    for i, arm in enumerate(ARMS):
        by_family = defaultdict(list)
        for r in records[arm]:
            by_family[(r["checkpoint_seed"], *(r[f] for f in FAMILY))].append(
                r["completion_status"] == "complete")
        for e in (.7, .9, .99):
            points = []
            for j, r in enumerate(x for x in diagnostics[arm] if x["eccentricity"] == e):
                outcomes = by_family[(r["checkpoint_seed"], *(r[f] for f in FAMILY))]
                assert len(outcomes) == 2 and outcomes[0] == outcomes[1]
                family_outcomes["complete" if outcomes[0] else "failed"] += 1
                points.append((3 * i + (.7, .9, .99).index(e)
                               + (((j * 17) % 61) / 60 - .5) * .42,
                               r["spectral_radius_maximum"]))
            lines.append(plot(points, f"only marks,mark={MARKS[arm]},mark size=1.5pt,"
                              f"color={COLORS[arm]}" + (",forget plot" if e != .7 else "")))
            if e == .7:
                lines.append(legend(LABELS[arm]))
    lines.append(plot([(-.5, 1), (8.5, 1)], "black,densely dashed,mark=none,forget plot"))

    lines.append(r"\nextgroupplot[title={(c) High-eccentricity monitor scale},"
                 r"ymode=log,xmin=-0.6,xmax=3.6,xtick={0,1,2,3},"
                 r"xticklabels={Unfactored,Factored .9,Factored .99,Manual},"
                 r"xticklabel style={font=\scriptsize},"
                 r"xlabel={Training recipe; deployment $e=0.99$},"
                 r"ylabel={Periapsis $g/r$}]")
    for i, arm in enumerate(ARMS):
        selected = [r for r in diagnostics[arm] if r["eccentricity"] == .99]
        assert len(selected) == 60
        lines.append(plot([(i + (((j * 17) % 61) / 60 - .5) * .42,
                            r["monitor_over_radius_at_periapsis"])
                           for j, r in enumerate(selected)],
                          f"only marks,mark={MARKS[arm]},mark size=1.6pt,color={COLORS[arm]},forget plot"))
    lines.append(plot([(-.6, 1), (3.6, 1)], "gray,densely dashed,mark=none,forget plot"))
    lines.append(plot([(3, 1)], "only marks,mark=square,mark size=2.5pt,color=orange!85!black,forget plot"))

    lines.append(r"\nextgroupplot[title={(d) Completed high-eccentricity runs},"
                 r"xmode=log,ymode=log,xlabel={Field-call count per period},"
                 r"ylabel={Final-state closure error},title style={yshift=1.05cm},"
                 r"legend style={at={(0.5,1.02)},anchor=south}]")
    accuracy_counts = {}
    for arm in (*ARMS, "manual_r"):
        selected = [r for r in records[arm] if r["eccentricity"] == .99
                    and r["completion_status"] == "complete"]
        accuracy_counts[arm] = len(selected)
        if not selected:
            continue
        lines.append(plot([(r["vector_field_evaluations"], r["final_state_closure_error"])
                           for r in selected],
                          f"only marks,mark={MARKS[arm]},mark size=1.7pt,color={COLORS[arm]}"))
        lines.append(legend(LABELS[arm]))
    assert accuracy_counts == {"invariant_e099": 0, "sundman_e09": 120,
                              "sundman_e099": 120, "manual_r": 24}
    finish(lines, "fig_factorisation_new", {
        "source_sha256": hashes, "completion": completions,
        "diagnostic_families": 540, "family_outcomes": dict(family_outcomes),
        "accuracy_points": accuracy_counts,
        "scope": "Complete recipes with differing initialization/training; common h=.04, not matched cost.",
        "diagnostic_unit": "One point per family, not duplicate points for the two rollout phases.",
        "manual_high_e_scale": "Exactly g/r=1; one reference marker, not a learned ensemble.",
    })


if __name__ == "__main__":
    mechanism()
    structure()
