# Experiment and evidence map

Internal AC identifiers preserve artifact paths; they are not manuscript
section, figure or table numbers.

| Scientific question | Principal saved inputs | Driver / audit |
| --- | --- | --- |
| Baseline deployment reliability (Section 5.1) | `results/expAC9/` | `expAC9_end_to_end_monitor_generalisation.py`; `new/audit_section_5_1.py` |
| Retrospective contraction diagnosis and step intervention (Section 5.2) | `results/expAC19/`; `results/returned_endpoint_review/verified/` | `expAC19_periapsis_contraction_diagnostic.py`; `new/audit_section_5_2.py` |
| Deployment-domain training and factored recipes (Section 5.3) | `results/expAC20/`; verified endpoint records | `expAC20_deployment_domain_training.py`; `new/audit_section_5_3.py` |
| Accuracy/work and applicability (Section 5.4) | `results/expAC10/`, `expAC13/`, `expAC16/`, `expAC18/` | `new/audit_section_5_4.py` |
| Corrected conditioning association (SI) | `results/expAC14/` | `expAC14_corrected_conditioning_audit.py` |
| Solver/architecture control (SI) | `results/expAC8/` | `expAC8_monitor_architecture_solver_factorial.py` |
| Observed versus enforced returned residual | `results/returned_endpoint_review/` and its `verified/` successor | `new/review_returned_endpoints.py`; `new/replay_verified_endpoints.py` |
| Implementation/refinement and long-run controls | `results/expAC0/` through `expAC7/` | Corresponding corrected AC drivers |

All experiment filenames above live under `experiments/` unless prefixed
with `new/`. Detailed settings are in the saved preregistration/training files.
The older `results/experiment_ledger.md` supplies historical context; its
numbers must be read with the later returned-residual correction.

## Corrections that must remain visible

- **AC19 intervention:** residual-enforced deployment completes 150/252;
  151/252 in the parent record is superseded. Parent records remain unchanged.
- **AC20:** factored recipes complete 360/360 each; expanded unfactored
  training completes 240/360. Multiple recipe choices changed, so this is
  not an isolated causal test of factorisation.
- **AC11:** its original conditioning operator is historical. AC14 uses the
  corrected operator and reports the finite observed association; it does
  not establish a general early-warning rule.
- **AC17:** superseded by AC18's corrected learned-H acceptance screen.
  AC18 accepts none of five learned Hamiltonians and has no downstream
  deployment records.
- **expE2:** only the checkpoint/frozen dependencies required by corrected
  controls are selected. Invalid historical known-H trajectories are not
  evidence for the current manuscript.

The reference-orbit contraction diagnostic is retrospective. A value below
one is not a convergence certificate, and a value above one is not a
necessary failure condition. Accuracy/work summaries retain the recorded
completion subsets, solver checks and reference limitations; no matched-cost
advantage is asserted.
