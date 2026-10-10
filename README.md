# PASNet: learned time transformations for stiff Hamiltonian systems

Research code and saved evidence for **Learning Time Transformations for Stiff
Hamiltonian Systems: Deployment Reliability and Applicability Boundaries**.

Authors: Jiangshan Wang, Shuxiang Xu and Shuai Zhu. Correspondence: Shuai Zhu
(zhushuaisjtu@gmail.com).

We call this implementation **PASNet (Poincaré-Adaptive Symplectic Network)**.
It combines a learned positive time monitor with the Poincaré extended
Hamiltonian formulation and implicit midpoint integration. This study examines
deployment reliability and applicability boundaries. The experiments do not
establish a matched-cost advantage or a general solver-convergence theorem.

## Quick start

Use Python 3.12. CPU execution is sufficient for the validation below.

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on Linux/macOS, then run:

```text
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python reproduce.py
```

The default command restores two losslessly compressed JSON files, verifies
SHA256 hashes, runs **83 numerical/integrity tests**, and executes the four
Section 5 saved-record audits. It writes a report to
`validation/VALIDATION_REPORT.json`. It does **not** retrain models or rerun
the full numerical experiments. The original Linux training environment is
recorded separately in `requirements-lock.txt`; these review pins do not claim
bitwise-equivalent retraining across platforms.

```text
python reproduce.py --verify-only
python reproduce.py --restore-only
```

`--verify-only` checks packaged inputs without extraction. `--restore-only`
extracts and verifies the raw records without testing. Large records are
stored as `results/expAC11/raw_results.json.gz` and
`results/expAC14/raw_results.json.gz`; the restored JSON files are ignored by
Git. Raw bytes, values and historical hashes are preserved.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Hamiltonian systems, PASNet, midpoint solvers and diagnostics |
| `experiments/` | Numbered corrected experiment drivers, training, protocol registration and analysis |
| `tests/` | Numerical correctness and saved-artifact integrity tests |
| `results/` | Protocols, checkpoints, raw records, summaries and correction history |
| `new/` | Returned-residual review, Section 5 audits, current scientific figure generators and manuscript/SI source snapshots |
| `theory/` | Corrected auxiliary analysis notes |
| `INPUT_MANIFEST.json` | Original input hashes and compression mapping |
| `docs/EXPERIMENT_MAP.md` | Current result sources and superseded-artifact warnings |

Original paths and bytes are retained because frozen protocols hash-link code
and checkpoints. `new/` is a historical directory name, not a second method.
The current manuscript sources are scientific documentation for the evidence
audits; old manuscript versions, cover letters, editing histories and downloaded
literature are excluded. This repository is independent of the earlier
directional-barrier paper.

## Results and scope

The main baseline source is AC9; contraction diagnostics and step interventions
are AC19, and deployment-domain/factored monitors are AC20. The verified
returned-residual successor takes precedence for the intervention completion
count: **150/252**, not the historical **151/252**. The completed factored
recipes change several training choices, so they do not isolate a causal
factorisation effect. AC10 supports accuracy/work comparisons with their
completion and reference limitations; AC13 covers the finite tested modal and
Duffing configurations. See `docs/EXPERIMENT_MAP.md` for details.

Some superseded artifacts remain for correction traceability and source-hash
tests. In particular, AC11 conditioning measurements were superseded by AC14,
and AC17 learned-H results by AC18. Their presence does not validate the
withdrawn claims. The historical expE2 checkpoint is retained for corrected
controls; the invalid known-H trajectories from that family are not included.

The numerical suite excludes one explicitly reported historical provenance
test requiring a file outside this study. Three additional tests tied solely
to old manuscript claim coverage are not distributed. Thus the code package
runs 83 tests, compared with 86 in the larger portable manuscript archive.
All other numerical tests are retained unchanged. The four Section 5 audits
validate saved records, not fresh experimental reproducibility or manuscript
style/submission compliance.

## Fresh experiments and figures

For a small new correctness run, after restoring the inputs:

```text
python experiments/expAC0_correctness_smoke.py
```

Experiment drivers can write into their historical result locations. Run fresh
training/replays in a separate checkout or use a driver's output argument where
supported. Keep the published evidence unchanged. Protocol-registration scripts
record new hashes; they should not overwrite sealed protocols for validation.
Full training and high-eccentricity replays are separate, potentially expensive
operations; `reproduce.py` never starts them.

Current figure generators are `new/make_deployment_figures.py` and
`new/make_section_5_4_figure.py` (PGFPlots/LaTeX sources),
`new/make_memory_figure.py` (Matplotlib), and
`experiments/analyze_expAC9.py` (baseline figure). The archived figures are
included. Figure regeneration requires the audited records and, for PGFPlots,
an existing LaTeX installation. LaTeX is not required for numerical validation.

## Release status and reuse

This is a **local, unpublished repository candidate**. No GitHub URL, release
tag or archive DOI has been assigned. A code/data/checkpoint license has not
yet been selected by the authors; no open-source permission is implied by
this preparation. Before public release, choose the relevant licenses and
record the final version/DOI in the article's code and data availability
statement. See `docs/PUBLISHING.md` for the upload steps.
