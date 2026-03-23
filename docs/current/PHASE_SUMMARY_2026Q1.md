# Phase Summary 2026Q1

## Scope

This summary captures the current state of the project after the first full mechanism-validation phase.

It separates:

- mechanisms that are now supported by experiments
- mechanisms that are implemented but not yet independently validated
- mechanisms that were explored but did not outperform the current best runtime path

## Current Position

The project is no longer only a research prototype.

It now has:

- a stable runtime
- a structured session and turn log
- typed artifact memory
- hypergraph-aware retrieval
- conflict-aware debugging retrieval
- continuity-aware evaluation
- real MiniMax-backed mechanism validation on selected experiment tracks
- first procedural-memory structures for reusable checklists, templates, and playbooks

The strongest validated retrieval path remains:

- `hyperedge_expansion` for general structured recovery
- `conflict_hyperedge_aware` for conflict-heavy debugging and root-cause queries

The strongest runtime continuity story is now:

- dedicated continuity regression is fully green on the local deterministic path
- product-style regression now covers six handoff / review / closure tasks
- runtime clearly beats the recent-history baseline on:
  - task continuation
  - repeated-work avoidance
- current remaining weakness is narrower:
  - product decision continuity is only partially separated from the baseline

## Validated Mechanisms

### 1. Competition

Status:

- validated

Evidence:

- [\_experiment_competition_llm.json](F:/hypergraph_bistability/_experiment_competition_llm.json)

Conclusion:

- competition reduces irrelevant context
- competition improves precision over direct retrieval
- competition preserves recall

Interpretation:

- pre-focus gating is a real mechanism, not only a design preference

### 2. Hyperedge-Aware Expansion

Status:

- validated
- promoted into formal runtime

Evidence:

- [\_experiment_associative_expansion_llm.json](F:/hypergraph_bistability/_experiment_associative_expansion_llm.json)

Conclusion:

- `hyperedge_expansion` preserves recall and response recall
- it reduces irrelevant context relative to `parent_expansion`
- it is the current best structural retrieval strategy for chain recovery

Interpretation:

- hyperedges are not just storage metadata
- they now provide measurable retrieval value

### 3. Confidence-Aware Retrieval

Status:

- validated locally
- partially promoted into formal runtime

Evidence:

- [\_experiment_confidence_tags_runtime_promoted.json](F:/hypergraph_bistability/_experiment_confidence_tags_runtime_promoted.json)

Conclusion:

- confidence tags improve filtering of contradicted or weak hypotheses
- main gain is in precision and irrelevant-context suppression
- response-level gain is weaker than retrieval-level gain

Interpretation:

- confidence is useful, but currently acts mostly as a local ranking modifier

### 4. Conflict Hyperedges

Status:

- validated locally
- validated on real MiniMax path
- partially promoted into formal debugging retrieval

Evidence:

- [\_experiment_conflict_links_v4.json](F:/hypergraph_bistability/_experiment_conflict_links_v4.json)
- [\_experiment_conflict_links_llm.json](F:/hypergraph_bistability/_experiment_conflict_links_llm.json)

Conclusion:

- `conflict_hyperedge_aware` outperforms single-edge contradiction handling
- conflict units improve recall, precision, and irrelevant-context suppression
- real MiniMax-backed conflict experiment also holds:
  - recall `1.000`
  - precision proxy `1.000`
  - irrelevant context rate `0.000`
  - response recall `1.000`

Interpretation:

- conflict should be treated as a hypergraph-level unit, not just a pairwise edge

## Implemented But Not Independently Proven

### 1. Hyperedge State

Status:

- implemented
- not independently validated

Current runtime support:

- `active`
- `paused`
- `resolved`
- `superseded`
- `conflicted`

Current conclusion:

- state is present in `hypergraph_view`
- state affects retrieval scoring
- current dedicated experiments did not isolate a measurable advantage

Why not yet validated:

- current scenarios are still dominated by text match and structural relation
- state has not yet become the decisive signal

### 2. Contradiction Linkage as a Primitive

Status:

- implemented
- structurally useful

Current conclusion:

- single `contradicts` edges are a useful primitive
- but they are weaker than `conflict_hyperedges`

Interpretation:

- contradiction edges should remain in the system
- but not as the final conflict abstraction

### 3. Decision Residue

Status:

- implemented
- partially connected to formal decision-continuity retrieval
- not independently superior to current best runtime

Current support:

- `decision`, `rationale`, `constraint`, and `alternative` are first-class node types
- decision residue relations are persisted:
  - `rationale_for`
  - `constrains`
  - `alternative_to`
- `decision_residues` are visible in `hypergraph_view`

Evidence:

- [\_experiment_decision_residue.json](F:/hypergraph_bistability/_experiment_decision_residue.json)

Conclusion:

- decision residue is clearly valuable relative to baseline
- it improves decision-continuity recovery over recent-history-only baselines
- it does not yet outperform the current formal runtime path

Interpretation:

- decision residue is a useful structural extension
- but the current runtime already recovers most of this signal through existing hyperedge expansion and response contracts
- this mechanism likely needs harder decision-progress scenarios before it can show independent gain

### 4. Task Phase / Progress State

Status:

- implemented
- visible in writes, retrieved detail, and `hypergraph_view`
- not independently superior to current best runtime

Current support:

- `task_phase` is now inferred at write time
- current phase set:
  - `analysis`
  - `decision`
  - `implementation`
  - `verification`
  - `closure`
- hyperedges expose `member_task_phases` and `phase_summary`

Evidence:

- [\_experiment_phase_progress.json](F:/hypergraph_bistability/_experiment_phase_progress.json)

Conclusion:

- phase-aware structure is valuable relative to baseline
- it helps separate implementation- and verification-oriented recovery from phase-blind baselines
- it does not yet outperform the current formal runtime path

Interpretation:

- progress-state structure is now available as a first-class signal
- but current runtime behavior already captures much of the same information through existing structured retrieval and continuity contracts
- future gains will likely require harder progress-state scenarios, not more local score tuning

### 5. Procedural Memory V1

Status:

- implemented
- independently validated against a procedure-aware mechanism slice
- conservatively integrated into the formal runtime

Current support:

- procedure-like content is typed as:
  - `playbook`
  - `checklist`
  - `template`
  - `procedure`
- procedure-like content is grouped into `procedure_hyperedge`
- procedure-like writes carry:
  - `procedure_type`
  - `reusability_class`
- `procedure_residues` are visible in `hypergraph_view`

Evidence:

- [\_experiment_procedure_memory.json](F:/hypergraph_bistability/_experiment_procedure_memory.json)
- [\_product_regression_after_procedure_runtime.json](F:/hypergraph_bistability/_product_regression_after_procedure_runtime.json)
- [\_long_task_regression_after_procedure_runtime.json](F:/hypergraph_bistability/_long_task_regression_after_procedure_runtime.json)
- [\_product_regression_v12.json](F:/hypergraph_bistability/_product_regression_v12.json)
- [\_product_regression_v13.json](F:/hypergraph_bistability/_product_regression_v13.json)
- [\_long_task_regression_v7.json](F:/hypergraph_bistability/_long_task_regression_v7.json)
- [\_long_task_regression_v8.json](F:/hypergraph_bistability/_long_task_regression_v8.json)
- [\_llm_product_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_product_regression_minimax_one_click.json)
- [\_llm_long_task_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_long_task_regression_minimax_one_click.json)

Conclusion:

- procedural memory now has a first-class schema path
- it fills a real missing layer between episodic work-state memory and reusable process knowledge
- procedure-aware retrieval clearly outperforms procedure-blind retrieval on checklist/template/closeout scenarios
- conservative runtime integration does not regress current product or long-task local regressions

Interpretation:

- procedural memory is now a supported data-organization primitive
- it has crossed from schema-only support into a validated and runtime-relevant layer
- the practical next step is to keep procedure-aware product/long-task regressions inside the formal runtime gate
- real MiniMax-backed product and long-task practical regressions now also run cleanly on the current procedure-heavy gate
- the next real-LLM step should stay on harder practical resume chains, not on adding many more procedure types first

## Explored But Not Better Than Current Best Path

### 1. Two-Stage + Conflict-Aware Retrieval

Status:

- implemented experimentally
- not superior to current best one-stage conflict-aware retrieval

Evidence:

- [\_experiment_two_stage_conflict_v2.json](F:/hypergraph_bistability/_experiment_two_stage_conflict_v2.json)

Conclusion:

- it beats single-edge contradiction handling
- it does not beat `conflict_hyperedge_aware`

Mechanism interpretation:

- current one-stage conflict-aware retrieval already captures the dominant structural signal
- the second stage did not introduce enough new discriminative information
- it mostly re-ranked already-correct candidates

## Product-Style Regression

Status:

- implemented
- now covers six multi-turn product tasks

Current product scenarios:

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

Latest evidence:

- [\_product_regression_v13.json](F:/hypergraph_bistability/_product_regression_v13.json)
- [\_llm_product_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_product_regression_minimax_one_click.json)

Current runtime summary:

- task continuation `1.000`
- blocker preservation `1.000`
- decision continuity `1.000`
- procedure continuity `1.000`
- repeated-work avoidance `1.000`

Current baseline summary:

- task continuation `0.000`
- blocker preservation `1.000`
- decision continuity `0.750`
- procedure continuity `0.750`
- repeated-work avoidance `0.000`

Interpretation:

- current runtime is already strong at carrying active work forward
- it is also strong at not restarting or asking the user to restate the task
- product-style decision continuity is now separated from the baseline
- product-style procedure continuity is also now separated from the baseline
- the key improvement came from letting decision queries explicitly admit `decision_hyperedge` members as retrieval candidates
- procedure-aware product continuation now also benefits from preserving dedicated `procedure_hyperedge` units instead of collapsing them into parent task/change units
- a harder local product stress expansion now exists, but some product-side procedure scenarios still leak to recent-history baselines more than the long-task procedure scenarios do
- the real MiniMax-backed product regression also now runs with:
  - retrieval recall `1.000`
  - response recall `0.963`
  - provider error count `0`
  - runtime task/blocker/decision/procedure continuity all `1.000`
- this confirms that `decision residue` becomes useful when the query explicitly asks for:
  - commitments
  - rejected alternatives
  - active constraints
  - scope guards

Current long-task practical continuation also now includes:

- `procedure_incident_closeout_replay`
- `procedure_incident_handoff_closeout_replay`

Latest long-task evidence:

- [\_long_task_regression_v8.json](F:/hypergraph_bistability/_long_task_regression_v8.json)
- [\_llm_long_task_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_long_task_regression_minimax_one_click.json)

Current long-task runtime summary:

- task continuation `1.000`
- blocker preservation `1.000`
- decision continuity `1.000`
- procedure continuity `1.000`
- repeated-work avoidance `1.000`

Current long-task baseline summary:

- task continuation `0.000`
- blocker preservation `0.500`
- decision continuity `0.333`
- procedure continuity `0.000`
- repeated-work avoidance `0.000`

Real MiniMax-backed long-task runtime summary:

- retrieval recall `1.000`
- response recall `1.000`
- provider error count `0`
- task continuation `1.000`
- blocker preservation `1.000`
- decision continuity `1.000`
- procedure continuity `1.000`
- repeated-work avoidance `1.000`

Practical decision:

- do not prioritize this path further for now

### 2. Uncertainty + Conflict Combined Retrieval

Status:

- implemented experimentally
- no measurable gain over current conflict-aware baseline

Evidence:

- [\_experiment_uncertainty_conflict.json](F:/hypergraph_bistability/_experiment_uncertainty_conflict.json)

Conclusion:

- `aware_vs_conflict_only` currently shows no gain
- `aware_vs_blind` also shows no gain

Mechanism interpretation:

- current confidence tags are attached to individual writes
- they are not yet a strong enough conflict-unit-level signal
- therefore they do not add new information beyond the existing conflict structure

Practical decision:

- do not prioritize this path further in the current form

## Best Current Runtime Story

The current best runtime story is:

1. competition reduces noise before focus
2. hyperedge-aware expansion restores structured context without recall loss
3. conflict hyperedges outperform single contradiction links in debugging recovery
4. confidence tags help as secondary ranking modifiers

This means the most credible current system identity is:

- hypergraph-structured agent memory with conflict-aware debugging retrieval
- plus structured continuity support for decisions and task progress, even where those newer structures have not yet exceeded the strongest runtime path

## Recommended Next Priorities

### Priority 1

- formalize this phase in docs and keep the validated path stable

### Priority 2

- design harder real-LLM stress cases that force new information sources, instead of only deeper heuristic layering

### Priority 3

- pursue only mechanisms that introduce genuinely new structure, such as:
  - richer hyperedge state
  - harder task-phase / progress-state scenarios
  - stronger decision-progress residue scenarios
  - continuity metrics beyond recall

## What To Avoid Next

- repeated small score tuning on already-strong retrieval paths
- continued investment in two-stage conflict retrieval without a new structural signal
- continued investment in uncertainty+conflict without conflict-unit-level uncertainty structure

## Short Verdict

This phase succeeded.

The project now has multiple experimentally supported mechanisms, and the strongest ones are already reflected in the formal runtime.

The main lesson from this phase is:

- new gains now come from new structure, not from more ranking heuristics layered onto the same structure

## Continuity Evaluation Outcome

This phase also established the first usable continuity-evaluation layer.

Current documentation:

- [CONTINUITY_METRICS_V1.md](F:/hypergraph_bistability/docs/CONTINUITY_METRICS_V1.md)

Current dedicated continuity regression set:

- `debugging_resume_with_preference`
- `blocker_resume_no_reexplaining`
- `task_continuity`
- `decision_resume_after_interruption`
- `artifact_chain_resume`
- `plan_resume_without_restart`
- `contradiction_link_resume`
- `conflict_unit_dominance`

Current continuity regression snapshot:

- runtime
  - `task_continuation = 1.000`
  - `blocker_preservation = 1.000`
  - `decision_continuity = 1.000`
  - `repeated_work_avoidance_proxy = 1.000`

- baseline
  - `task_continuation = 0.0`
  - `blocker_preservation = 0.875`
  - `decision_continuity = 1.000`
  - `repeated_work_avoidance_proxy = 0.0`

Interpretation:

- the runtime now demonstrates measurable continuity gains beyond raw retrieval recall
- dedicated continuity regression is now strong enough to serve as a default runtime gate
- future runtime changes should be expected not to regress continuity, not only recall/precision

## Latest Mainline Update

- `stable_v1` remains the formal runtime.
- the remaining real MiniMax-backed product regression response failure in `debug_fix_verify_close_loop` was closed by tightening the closeout response contract for `before closing` / incident-close queries
- the fix was intentionally narrow:
  - keep retrieval unchanged
  - preserve the verified fix and reconnect-ordering recall
  - force the response layer to state the closeout condition explicitly when the retrieved evidence already supports it

Latest real MiniMax-backed product regression summary:

- retrieval recall `1.000`
- response recall `1.000`
- provider errors `0`
- runtime task continuation `1.000`
- runtime blocker preservation `1.000`
- runtime decision continuity `1.000`
- runtime procedure continuity `1.000`
- runtime repeated-work avoidance `1.000`

Interpretation:

- the current product-side procedure-aware gate is now clean on the real MiniMax path
- the remaining work should shift from patching this specific closeout gap to expanding harder practical stress cases and sidecar exploration

## Latest Sidecar Update

- added a dedicated practical sidecar suite for harder paraphrase-heavy handoff/closeout packet queries
- current sidecar set:
  - `sidecar_incident_handoff_bundle_paraphrase`
- this suite is intentionally not part of the formal `product` or `long-task` gate yet

Latest local sidecar snapshot:

- output: [\_practical_sidecar_regression_v6.json](F:/hypergraph_bistability/_practical_sidecar_regression_v6.json)
- runtime task continuation `1.000`
- baseline task continuation `0.000`
- runtime decision continuity `1.000`
- baseline decision continuity `0.667`
- runtime procedure continuity `1.000`
- baseline procedure continuity `0.000`
- runtime repeated-work avoidance `1.000`
- baseline repeated-work avoidance `0.000`

Interpretation:

- the sidecar suite now cleanly separates from the baseline on `procedure_continuity` and `repeated_work_avoidance_proxy` without disturbing the formal product gate
- the remaining sidecar handoff-bundle variant is still useful, but it is conflict-heavy rather than procedure-residue-heavy, so it remains exploratory rather than promote-ready

## Latest Robustness Check

- added a dedicated practical robustness sidecar to test whether the runtime still works after removing explicit trigger words such as `handoff`, `close`, and `packet`
- current robustness set:
  - `robustness_incident_story_bundle`
  - `robustness_release_scope_bundle`

Latest local robustness snapshot:

- output: [\_practical_robustness_regression_v4.json](F:/hypergraph_bistability/_practical_robustness_regression_v4.json)
- runtime task continuation `1.000`
- baseline task continuation `0.000`
- runtime decision continuity `1.000`
- baseline decision continuity `1.000`
- runtime repeated-work avoidance `1.000`
- baseline repeated-work avoidance `0.000`

Readout:

- this is evidence that the current runtime is not only passing because of explicit `handoff` / `close` / `packet` wording
- after treating `story still holds / proof points` as verification-style queries, the runtime now also recovers the verified fix / staging evidence path in this de-triggered suite
- the robustness suite should remain a `sidecar` guard against local optimization rather than a promote-ready formal gate, but it is now substantially healthier than the first pass

## Latest Promote Decision

- `promote`: `sidecar_incident_close_packet_paraphrase`
- reason:
  - runtime recall / response / continuity were already clean
  - baseline still failed materially
  - the scenario preserves a stable `procedure_residue`
  - the scenario is tightly aligned with the mainline incident closeout / procedure-continuity objective

Latest product gate snapshot after promotion:

- output: [\_product_regression_v22.json](F:/hypergraph_bistability/_product_regression_v22.json)
- runtime task continuation `1.000`
- baseline task continuation `0.000`
- runtime decision continuity `1.000`
- baseline decision continuity `0.556`
- runtime procedure continuity `1.000`
- baseline procedure continuity `0.200`
- runtime repeated-work avoidance `1.000`
- baseline repeated-work avoidance `0.000`

Interpretation:

- the promoted close-packet case tightened the formal product gate without regressing the runtime path
- the remaining paraphrase-heavy packet / robustness cases should stay `sidecar` until they either gain stable structure signals or prove their value over multiple rounds

## Latest Promote Decision 2

- `promote`: `sidecar_release_packet_followthrough_paraphrase`
- reason:
  - runtime recall / response / continuity were clean
  - baseline still failed materially
  - after fixing procedure inference on explicit packet procedures, the scenario now yields a stable `procedure_residue`
  - the scenario is aligned with the mainline release review / packet continuity objective

Latest product gate snapshot after second promotion:

- output: [\_product_regression_v24.json](F:/hypergraph_bistability/_product_regression_v24.json)
- runtime task continuation `1.000`
- baseline task continuation `0.000`
- runtime decision continuity `1.000`
- baseline decision continuity `0.500`
- runtime procedure continuity `1.000`
- baseline procedure continuity `0.167`
- runtime repeated-work avoidance `1.000`
- baseline repeated-work avoidance `0.000`

Interpretation:

- the formal product gate now includes both promoted packet-paraphrase procedure cases
- the remaining candidate with strong signal is `sidecar_incident_handoff_bundle_paraphrase`, but it is still better interpreted as a conflict-heavy sidecar than a formal procedure gate

## Latest Cross-Path Check

- after the two packet-paraphrase promotions, the real MiniMax-backed product regression still runs cleanly
- output: [\_llm_product_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_product_regression_minimax_one_click.json)
  - retrieval recall `1.000`
  - response recall `1.000`
  - provider errors `0`
  - runtime task / blocker / decision / procedure / repeated-work all `1.000`

- the long-task path also remains clean
- outputs:
  - [\_llm_long_task_regression_minimax_one_click.json](F:/hypergraph_bistability/_llm_long_task_regression_minimax_one_click.json)
  - [\_long_task_regression_v9.json](F:/hypergraph_bistability/_long_task_regression_v9.json)
  - real MiniMax long-task retrieval recall `1.000`
  - real MiniMax long-task response recall `1.000`
  - local long-task runtime task / blocker / decision / procedure / repeated-work all `1.000`

Interpretation:

- the promoted product-side procedure cases did not create a product-only gain that collapses on long-task or real LLM paths
- this is a stronger signal that the recent changes are actual runtime/structure gains rather than local benchmark fitting

## Latest Conflict Sidecar Decision

- the remaining non-promoted practical sidecar is now explicitly treated as a conflict-heavy sidecar:
  - `sidecar_incident_handoff_bundle_paraphrase`
- dedicated output: [\_conflict_sidecar_regression_v1.json](F:/hypergraph_bistability/_conflict_sidecar_regression_v1.json)
- current runtime readout:
  - task continuation `1.000`
  - blocker preservation `1.000`
  - decision continuity `1.000`
  - procedure continuity `0.000`
  - repeated-work avoidance `1.000`

Interpretation:

- this scenario is valuable because it preserves a dominant surviving hypothesis and supporting evidence through a conflict-heavy chain
- it should not be mistaken for a formal procedure gate, because its main structural signal is `conflict_heavy`, not `procedure_heavy`

## Latest Conflict Practical Line

- conflict-heavy practical regression is now tracked as an explicit line instead of only as a sidecar label
- dedicated scenarios:
  - `incident_root_cause_handoff`
  - `sidecar_incident_handoff_bundle_paraphrase`
- dedicated output: [\_conflict_regression_v1.json](F:/hypergraph_bistability/_conflict_regression_v1.json)
- new suite-level continuity field:
  - `conflict_continuity`

Current local conflict practical snapshot:

- runtime
  - task continuation `1.000`
  - blocker preservation `1.000`
  - decision continuity `1.000`
  - conflict continuity `1.000`
  - repeated-work avoidance `1.000`
- baseline
  - task continuation `0.000`
  - blocker preservation `1.000`
  - decision continuity `1.000`
  - conflict continuity `0.000`
  - repeated-work avoidance `0.000`

Interpretation:

- the project now has a clean practical line for conflict-heavy continuation that does not need to masquerade as a procedure gate
- `procedure_continuity` remains the right metric for procedure-heavy gates
- `conflict_continuity` is now the right metric for dominant-hypothesis / surviving-evidence continuation on conflict-heavy practical chains

## Latest Conflict Practical Expansion

- added a harder conflict-heavy practical scenario:
  - `incident_conflict_packet_resolution`
- this scenario asks for:
  - the surviving theory
  - the ruled-out theory
  - the proof points that still travel with the incident packet
- dedicated output: [\_conflict_regression_v2.json](F:/hypergraph_bistability/_conflict_regression_v2.json)

Current local conflict practical snapshot after expansion:

- runtime
  - task continuation `1.000`
  - blocker preservation `1.000`
  - decision continuity `1.000`
  - conflict continuity `1.000`
  - repeated-work avoidance `1.000`
- baseline
  - task continuation `0.000`
  - blocker preservation `1.000`
  - decision continuity `1.000`
  - conflict continuity `0.000`
  - repeated-work avoidance `0.000`

Interpretation:

- conflict-heavy practical regression now measures more than "remember the dominant theory"
- it now also checks whether the runtime can explicitly preserve the ruled-out theory boundary without collapsing back into a generic root-cause summary
- this is still mainline-compatible work because it strengthens the agent's work-state continuity on real incident handoff packets rather than opening a new mechanism track

## Latest Working-Set Query Layer

- added the first minimal working-set / query API on top of the current `hypergraph_view`
- current API surface:
  - `get_working_set()`
  - `query_current_task_state()`
  - `query_dominant_conflict()`
  - `query_decision_residue()`
  - `query_applicable_procedures()`
  - `query_handoff_bundle()`

Latest validation:

- `python -m pytest tests\test_agent_runtime.py tests\test_evals.py -q`
  - `131 passed`
- formal product regression remains clean:
  - [\_product_regression_v28.json](F:/hypergraph_bistability/_product_regression_v28.json)
- conflict practical regression remains clean:
  - [\_conflict_regression_v3.json](F:/hypergraph_bistability/_conflict_regression_v3.json)

Interpretation:

- this is the first concrete step from "strong internal structure" toward an inspectable agent work surface
- it does not replace the runtime path yet
- but it reduces the gap to practical infrastructure by making current task state, conflict state, procedure state, and handoff bundles queryable without inventing a new storage layer
