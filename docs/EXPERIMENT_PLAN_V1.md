# Experiment Plan v1

## Goal

Validate the core hypothesis behind the project:

1. memory items should compete before entering focus
2. salient items should seed associative expansion
3. mode-specific control should improve answer usefulness over generic retrieval

This plan defines the first three mechanism-level experiments. The purpose is not broad benchmark coverage. The purpose is to test whether the project's core claims are true.

## Continuity Layer

The project now also evaluates continuity, not only retrieval.

Current continuity metrics are defined in:

- [CONTINUITY_METRICS_V1.md](F:/hypergraph_bistability/docs/CONTINUITY_METRICS_V1.md)

Current suite-level continuity outputs include:

- `task_continuation`
- `blocker_preservation`
- `decision_continuity`
- `repeated_work_avoidance_proxy`

These metrics now appear:

- per scenario
- per suite
- in mechanism-experiment comparison summaries as deltas

This means future mechanism work should be judged on two layers:

- retrieval quality
- work continuity quality

## Product Regression Layer

The project now also maintains a small product-style regression set for realistic handoff and resume behavior.

Current scenarios:

- `release_hotfix_handoff`
- `debug_fix_verify_close_loop`
- `coding_review_commitment_chain`
- `incident_root_cause_handoff`
- `release_scope_guardrail_handoff`
- `review_scope_followup_chain`
- `procedure_release_handoff_chain`
- `procedure_review_handoff_chain`
- `procedure_release_gate_review_chain`
- `procedure_review_validation_handoff_chain`

Latest result:

- [\_product_regression_v12.json](F:/hypergraph_bistability/_product_regression_v12.json)
- [\_product_regression_v13.json](F:/hypergraph_bistability/_product_regression_v13.json)
- [\_llm_product_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_product_regression_minimax_one_click.json)

Current runtime summary:

- task continuation `1.000`
- blocker preservation `1.000`
- decision continuity `1.000`
- procedure continuity `1.000`
- repeated-work avoidance `1.000`

Current local product baseline summary after harder procedure-aware expansion:

- task continuation `0.000`
- blocker preservation `1.000`
- decision continuity `0.750`
- procedure continuity `0.750`
- repeated-work avoidance `0.000`

Current real MiniMax-backed product summary:

- retrieval recall `1.000`
- response recall `0.963`
- provider error count `0`
- task continuation `1.000`
- blocker preservation `1.000`
- decision continuity `1.000`
- procedure continuity `1.000`
- repeated-work avoidance `1.000`

Current interpretation:

- the formal runtime is already strong on product-style resume behavior
- the main remaining product gap is no longer generic continuity
- product-style decision continuity now also separates from the baseline
- product-style procedure continuity now also separates from the baseline
- the key runtime change was:
  - decision queries explicitly admit `decision_hyperedge` members into the retrieval candidate set
  - procedure queries preserve dedicated `procedure_hyperedge` units inside the formal runtime path

## Procedural Memory Layer

`Procedural Memory V1` is now a validated mechanism layer, not only a schema addition.

Current procedure-aware scenarios:

- `procedure_release_handoff`
- `procedure_review_template`
- `procedure_incident_closeout`

Key evidence:

- [\_experiment_procedure_memory.json](F:/hypergraph_bistability/_experiment_procedure_memory.json)
- [\_product_regression_after_procedure_runtime.json](F:/hypergraph_bistability/_product_regression_after_procedure_runtime.json)
- [\_long_task_regression_after_procedure_runtime.json](F:/hypergraph_bistability/_long_task_regression_after_procedure_runtime.json)
- [\_product_regression_v12.json](F:/hypergraph_bistability/_product_regression_v12.json)
- [\_long_task_regression_v7.json](F:/hypergraph_bistability/_long_task_regression_v7.json)
- [\_llm_product_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_product_regression_minimax_one_click.json)
- [\_llm_long_task_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_long_task_regression_minimax_one_click.json)

Current interpretation:

- procedure-aware retrieval adds independent value over a procedure-blind variant
- reusable checklists, templates, and closeout procedures now behave like first-class retrieval targets
- conservative promotion into the formal runtime does not regress product or long-task local regressions
- procedure-aware practical regressions are now part of the formal product/long-task gate, not only a side experiment
- `procedure_continuity` should now be tracked alongside task/blocker/decision continuity on procedure-heavy suites
- real MiniMax-backed product and long-task practical regressions now also run cleanly on this gate after preserving verification-phase evidence for handoff/closeout queries
- harder local product procedure scenarios still leave some recent-history leakage, so future product-side procedure stress should emphasize longer interruptions and weaker lexical overlap

## Experiment 1: Competition vs Direct Retrieval

### Hypothesis

Competition-driven selection improves context precision and response usefulness over direct retrieval without competition.

### Comparison Groups

- `recent_history_baseline`
- `direct_retrieval`
  - lexical/typed retrieval without competition gating
- `competition_retrieval`
  - current runtime behavior with activation-based working-memory selection

### Scenarios

- `preference_recall`
- `task_continuity`
- `context_switching`
- `layered_preferences`

### Metrics

- retrieval recall
- retrieval precision proxy
- response recall
- irrelevant context rate
- average retrieved item count

### Success Criterion

`competition_retrieval` should preserve recall while improving precision proxy and reducing irrelevant context.

### Code Surfaces

- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [turn_processor.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/turn_processor.py)
- new ablation runner in `src/hypergraph_bistability/experiments/`

## Experiment 2: Associative Expansion vs Single-Hit Retrieval

### Hypothesis

Associative expansion improves task-chain continuity compared with single-hit retrieval.

### Comparison Groups

- `single_hit`
  - only keep directly matched item(s)
- `parent_expansion`
  - existing parent-artifact expansion
- `hyperedge_expansion`
  - planned hyperedge-aware expansion

### Scenarios

- `debugging_resume_with_preference`
- `coding_agent_resume`
- `artifact_chain_resume`
- `artifact_relation_chain`

### Metrics

- retrieval recall
- response recall
- chain recall
- explicit step recall
- task continuity score

### Success Criterion

`parent_expansion` should beat `single_hit`, and future `hyperedge_expansion` should beat both on chain-oriented scenarios.

### Current Result

This experiment is now implemented and has a validated best variant.

Local result:

- `parent_expansion`
  - recall `1.000`
  - precision proxy `0.800`
  - irrelevant context rate `0.200`
  - response recall `1.000`
- `hyperedge_expansion`
  - recall `1.000`
  - precision proxy `1.000`
  - irrelevant context rate `0.000`
  - response recall `1.000`

Real MiniMax result:

- output file: [\_experiment_associative_expansion_llm.json](F:/hypergraph_bistability/_experiment_associative_expansion_llm.json)
- `hyperedge_vs_parent`
  - `recall_delta = 0.0`
  - `precision_proxy_delta = +0.2`
  - `irrelevant_context_rate_delta = -0.2`
  - `response_recall_delta = 0.0`

Interpretation:

- `single_hit` is too weak for chain-heavy work recovery
- `parent_expansion` restores chain recall but adds noise
- `hyperedge_expansion` preserves recall and response quality while removing irrelevant context

Current recommendation:

- treat `hyperedge_expansion` as the preferred Experiment 2 strategy
- use it as the structural base for future two-stage competitive retrieval

### Code Surfaces

- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)
- future `hypergraph_view` implementation

## Experiment 3: Mode-Specific Policy vs Generic Policy

### Hypothesis

Mode-specific response and retrieval policies improve response utilization without reducing retrieval recall.

### Comparison Groups

- `generic_mode`
  - one response contract and one retrieval policy for all tasks
- `mode_specific`
  - debugging/coding/planning/review-specific contracts and weighting

### Scenarios

- `task_continuity`
- `debugging_resume_with_preference`
- `coding_agent_resume`
- `artifact_relation_chain`

### Metrics

- retrieval recall
- response recall
- irrelevant context suppression
- explicit issue/task restatement rate

### Success Criterion

`mode_specific` should materially improve response recall in stress scenarios while preserving retrieval recall.

### Code Surfaces

- [context_assembler.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/runtime/context_assembler.py)
- [hypergraph_agent.py](F:/hypergraph_bistability/src/hypergraph_bistability/agent/hypergraph_agent.py)
- [retrieval_policy.py](F:/hypergraph_bistability/src/hypergraph_bistability/memory/policies/retrieval_policy.py)

## Experiment 4: Conflict Hyperedges vs Single Contradiction Links

### Hypothesis

Conflict hyperedges should outperform single-edge contradiction handling when multiple competing hypotheses remain active inside the same task unit.

### Comparison Groups

- `conflict_hyperedge_aware`
  - retrieval uses derived conflict units and favors the dominant remaining hypothesis
- `conflict_link_aware`
  - retrieval only uses `contradicts` edge targets
- `conflict_link_blind`
  - retrieval ignores contradiction structure
- `recent_history_baseline`

### Scenarios

- `contradiction_link_filtering`
- `conflict_unit_dominance`
- `contradiction_link_resume`

### Metrics

- retrieval recall
- retrieval precision proxy
- response recall
- irrelevant context rate

### Current Result

This experiment is now implemented and has a clean structural result.

Output file:

- [\_experiment_conflict_links_v4.json](F:/hypergraph_bistability/_experiment_conflict_links_v4.json)

Key comparison:

- `hyperedge_aware_vs_link_aware`
  - `recall_delta = +0.333`
  - `precision_proxy_delta = +0.333`
  - `irrelevant_context_rate_delta = -0.333`
  - `response_recall_delta = +0.143`

Interpretation:

- single `contradicts` edges are useful as a base mechanism
- conflict hyperedges are stronger when several competing hypotheses share one task/evidence unit
- conflict should be treated as a hypergraph-level structure, not only as pairwise negation

Real MiniMax result:

- output file: [\_experiment_conflict_links_llm.json](F:/hypergraph_bistability/_experiment_conflict_links_llm.json)
- `conflict_hyperedge_aware`
  - recall `1.000`
  - precision proxy `1.000`
  - irrelevant context rate `0.000`
  - response recall `1.000`
- `hyperedge_aware_vs_link_aware`
  - `recall_delta = +0.333`
  - `precision_proxy_delta = +0.333`
  - `irrelevant_context_rate_delta = -0.333`
  - `response_recall_delta = 0.0`

Interpretation:

- the structural advantage survives on the real MiniMax path
- response quality is also fully preserved
- this makes `conflict_hyperedges` one of the strongest validated mechanisms in the project

### Recommendation

- keep `contradicts` as the primitive relation
- treat `conflict_hyperedges` as the preferred continuation of that primitive
- use conflict units as the foundation for future conflict-aware runtime and debugging-mode retrieval

## Execution Order

### Batch A

- build experiment runner that supports ablation modes
- keep existing eval suite untouched
- emit JSON results per experiment

### Batch B

- implement Experiment 1
- validate competition effect

### Batch C

- implement Experiment 2
- validate associative expansion effect
- completed
- current best variant: `hyperedge_expansion`

### Batch D

- implement Experiment 3
- validate mode-specific effect

## Output Files

Suggested output naming:

- `_experiment_competition.json`
- `_experiment_associative_expansion.json`
- `_experiment_mode_specific.json`

## Immediate Recommendation

Start with Experiment 1.

Reason:

- it is the cleanest test of the core claim
- it can be implemented with the least schema change
- it gives a direct answer to whether the "competition before focus" idea has real value

## Current Continuity Baseline

The eval suite now includes a dedicated continuity regression set:

- `debugging_resume_with_preference`
- `blocker_resume_no_reexplaining`
- `task_continuity`
- `decision_resume_after_interruption`
- `artifact_chain_resume`
- `plan_resume_without_restart`
- `contradiction_link_resume`
- `conflict_unit_dominance`

Current dedicated continuity regression snapshot:

- runtime
  - `task_continuation = 1.000`
  - `blocker_preservation = 1.000`
  - `decision_continuity = 0.625`
  - `repeated_work_avoidance_proxy = 0.750`

- baseline
  - `task_continuation = 0.0`
  - `blocker_preservation = 0.875`
  - `decision_continuity = 0.625`
  - `repeated_work_avoidance_proxy = 0.0`

Interpretation:

- the runtime is now measurably better than the baseline on continuation behavior itself
- `decision_continuity` remains the weakest of the dedicated continuity dimensions and should be treated as a future improvement target
- this regression set should be treated as a required evaluation layer for future retrieval and runtime changes

## Latest Promote Result

- `promote`: tighten the closeout response contract for `debug_fix_verify_close_loop` instead of changing retrieval
- root cause:
  - the retrieved evidence already contained `Verified:` closeout support
  - the response post-process still keyed too narrowly on `close`, so `before closing the incident` did not always force an explicit closeout conclusion
- applied fix:
  - keep retrieval untouched
  - treat `before closing` / verified closeout evidence as sufficient to append the explicit `ready to close` conclusion when the response omits it

Latest real MiniMax-backed product result:

- retrieval recall `1.000`
- response recall `1.000`
- provider errors `0`
- runtime task continuation `1.000`
- runtime blocker preservation `1.000`
- runtime decision continuity `1.000`
- runtime procedure continuity `1.000`
- runtime repeated-work avoidance `1.000`

Next exploration candidates:

- sidecar paraphrase-heavy incident closeout stress
- sidecar handoff-bundle responses that jointly express root cause, fix, verification, and closure state
- harder product practical scenarios that reduce recent-history lexical leakage further without weakening runtime continuity

## Latest Sidecar Result

- `sidecar`: practical paraphrase-heavy handoff/closeout packet suite
- implemented as a dedicated runner instead of extending the formal product gate immediately
- current scenarios:
  - `sidecar_incident_handoff_bundle_paraphrase`

Latest local result:

- output: [\_practical_sidecar_regression_v6.json](F:/hypergraph_bistability/_practical_sidecar_regression_v6.json)
- runtime task continuation `1.000`
- runtime decision continuity `1.000`
- runtime procedure continuity `1.000`
- baseline procedure continuity `0.000`
- runtime repeated-work avoidance `1.000`

Readout:

- the sidecar suite now runs cleanly on the runtime path while the baseline still fails by retrieval
- the handoff-bundle variant remains more conflict-unit-heavy than procedure-residue-heavy, which is informative and should stay exploratory for now
- the remaining handoff-bundle variant is still conflict-heavy rather than procedure-residue-heavy, so it remains exploratory

Next step:

- keep the remaining handoff-bundle variant as sidecar unless it develops a cleaner formal-gate interpretation than conflict-heavy recovery

## Latest Robustness Sidecar

- `sidecar`: de-triggered practical robustness suite
- purpose:
  - check whether current gains survive after removing explicit query triggers like `handoff`, `close`, and `packet`
- current scenarios:
  - `robustness_incident_story_bundle`
  - `robustness_release_scope_bundle`

Latest local result:

- output: [\_practical_robustness_regression_v4.json](F:/hypergraph_bistability/_practical_robustness_regression_v4.json)
- runtime task continuation `1.000`
- baseline task continuation `0.000`
- runtime decision continuity `1.000`
- runtime repeated-work avoidance `1.000`
- baseline repeated-work avoidance `0.000`

Readout:

- the runtime survives de-triggering better than the baseline, so the current system is not only phrase-fitting to explicit handoff/closeout wording
- both robustness cases now run cleanly on the runtime path
- the key fix was structural: treat `story still holds / fix still holds / proof points` as verification-style queries so verified evidence is no longer underweighted or filtered
- the suite should still stay `sidecar`; its value is as an anti-local-optimization check, not as a formal gate replacement

Next step:

- treat the robustness suite as a standing anti-local-optimization sidecar and use it to judge whether future improvements are real generalization or only phrase-specific fitting

## Latest Promote Decision

- `promote`: `sidecar_incident_close_packet_paraphrase`
- promoted into the formal `product` gate
- retained as a good promote candidate because it combines:
  - runtime recall `1.000`
  - runtime response recall `1.000`
  - runtime procedure continuity `1.000`
  - stable `procedure_residue`
  - clear baseline failure

Latest formal product result after promotion:

- output: [\_product_regression_v22.json](F:/hypergraph_bistability/_product_regression_v22.json)
- runtime task continuation `1.000`
- runtime decision continuity `1.000`
- runtime procedure continuity `1.000`
- baseline procedure continuity `0.200`
- runtime repeated-work avoidance `1.000`

Decision note:

- `sidecar_release_packet_followthrough_paraphrase` stays sidecar because it still lacks stable procedure residue
- `robustness_*` scenarios stay sidecar because their job is anti-local-optimization checking, not formal gate replacement

## Latest Promote Decision 2

- `promote`: `sidecar_release_packet_followthrough_paraphrase`
- trigger for promotion:
  - explicit packet procedures now infer as real procedure/template nodes instead of collapsing into `constraint`
  - the scenario now yields stable `procedure_residue`
  - runtime still runs cleanly while the baseline still fails by retrieval

Latest formal product result after second promotion:

- output: [\_product_regression_v24.json](F:/hypergraph_bistability/_product_regression_v24.json)
- runtime task continuation `1.000`
- runtime decision continuity `1.000`
- runtime procedure continuity `1.000`
- baseline procedure continuity `0.167`
- runtime repeated-work avoidance `1.000`

Decision note:

- the remaining `sidecar_incident_handoff_bundle_paraphrase` is still better treated as a conflict-heavy sidecar
- `robustness_*` scenarios remain anti-local-optimization sidecars rather than promote candidates

## Latest Cross-Path Validation

- the two promoted packet-paraphrase product cases were revalidated on the real MiniMax product path
- output: [\_llm_product_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_product_regression_minimax_one_click.json)
  - retrieval recall `1.000`
  - response recall `1.000`
  - runtime task / blocker / decision / procedure / repeated-work all `1.000`

- long-task remained clean at the same time
- outputs:
  - [\_llm_long_task_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_long_task_regression_minimax_one_click.json)
  - [\_long_task_regression_v9.json](F:/hypergraph_bistability/_long_task_regression_v9.json)
  - real MiniMax long-task retrieval recall `1.000`
  - real MiniMax long-task response recall `1.000`
  - local long-task runtime continuity all `1.000`

Readout:

- the recent promotions are not just local product-benchmark wins
- the current evidence supports keeping the promoted packet cases in the formal product gate while leaving the conflict-heavy / robustness suites as sidecars

## Latest Conflict Sidecar Position

- remaining practical sidecar:
  - `sidecar_incident_handoff_bundle_paraphrase`
- dedicated output: [\_conflict_sidecar_regression_v1.json](F:/hypergraph_bistability/_conflict_sidecar_regression_v1.json)
- explicit analysis tag:
  - `graph_profile = conflict_heavy`

Decision note:

- this scenario should be read as a conflict-heavy continuation stress, not as a failed procedure-gate candidate
- keep it as sidecar unless a future change makes it cleanly procedure-heavy rather than conflict-heavy

## Latest Conflict Practical Regression

- `conflict-heavy practical regression` is now tracked as its own line rather than being folded into the procedure gate story
- dedicated scenarios:
  - `incident_root_cause_handoff`
  - `sidecar_incident_handoff_bundle_paraphrase`
- dedicated output: [\_conflict_regression_v1.json](F:/hypergraph_bistability/_conflict_regression_v1.json)
- dedicated continuity field:
  - `conflict_continuity`

Current local result:

- runtime task continuation `1.000`
- runtime blocker preservation `1.000`
- runtime decision continuity `1.000`
- runtime conflict continuity `1.000`
- runtime repeated-work avoidance `1.000`
- baseline task continuation `0.000`
- baseline conflict continuity `0.000`
- baseline repeated-work avoidance `0.000`

Decision note:

- this line is worth keeping because it captures a real practical capability that `procedure_continuity` does not measure well
- it should stay parallel to the procedure-heavy product/long-task gates, not replace them

## Latest Conflict Practical Expansion

- added `incident_conflict_packet_resolution` into the conflict practical line
- purpose:
  - test whether the runtime can preserve both the surviving theory and the ruled-out theory when the user asks for an incident packet rather than a simple root-cause summary
- current output: [\_conflict_regression_v2.json](F:/hypergraph_bistability/_conflict_regression_v2.json)

Current local result after expansion:

- runtime task continuation `1.000`
- runtime decision continuity `1.000`
- runtime conflict continuity `1.000`
- runtime repeated-work avoidance `1.000`
- baseline task continuation `0.000`
- baseline conflict continuity `0.000`
- baseline repeated-work avoidance `0.000`

Decision note:

- this was a good mainline-serving expansion because it forced a real retrieval/response improvement around ruled-out theory boundaries
- it did not require a new runtime identity or a new research track

## Latest Working-Set Query Layer

- a minimal working-set / query layer is now implemented directly on top of `hypergraph_view`
- current API surface:
  - `get_working_set()`
  - `query_current_task_state()`
  - `query_dominant_conflict()`
  - `query_decision_residue()`
  - `query_applicable_procedures()`
  - `query_handoff_bundle()`

Current readout:

- this is not a separate mechanism experiment
- it is a practical infrastructure step that makes current task, conflict, decision, procedure, and handoff state inspectable through stable runtime queries
- current regression status stayed clean after adding it:
  - [\_product_regression_v28.json](F:/hypergraph_bistability/_product_regression_v28.json)
  - [\_conflict_regression_v3.json](F:/hypergraph_bistability/_conflict_regression_v3.json)
