"""Scripted multi-turn scenarios for memory evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class EvalTurn:
    """One user turn in a scripted evaluation scenario."""

    user_input: str
    expected_retrievals: List[str] = field(default_factory=list)
    expected_response_signals: List[str] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class EvalScenario:
    """A named multi-turn scenario with retrieval expectations."""

    name: str
    description: str
    tier: str
    turns: List[EvalTurn]
    tags: List[str] = field(default_factory=list)


DEFAULT_EVAL_SCENARIOS = [
    EvalScenario(
        name="preference_recall",
        description="The agent should recall explicit user preferences on a later turn.",
        tier="core",
        turns=[
            EvalTurn("Remember that I prefer concise answers and bullet lists."),
            EvalTurn(
                "How should you respond to me in future planning discussions?",
                expected_retrievals=["prefer concise answers", "bullet lists"],
                expected_response_signals=["concise", "bullet"],
            ),
        ],
    ),
    EvalScenario(
        name="task_continuity",
        description="The agent should surface an in-progress task when the user resumes it.",
        tier="core",
        turns=[
            EvalTurn("I am preparing a release checklist for version 2.1."),
            EvalTurn("The release checklist should include docs, migration notes, and rollback."),
            EvalTurn(
                "Pick up that release work again and tell me what matters most.",
                expected_retrievals=["release checklist", "migration notes", "rollback"],
                expected_response_signals=["release checklist", "migration notes", "rollback"],
            ),
        ],
    ),
    EvalScenario(
        name="context_switching",
        description="The agent should recover both the active task and a stable preference after topic shifts.",
        tier="core",
        turns=[
            EvalTurn("Remember that I prefer concise code reviews."),
            EvalTurn("I need help planning an onboarding checklist for a new backend engineer."),
            EvalTurn("Also, I want dinner ideas for tonight."),
            EvalTurn(
                "Back to the onboarding plan. Keep my preference in mind.",
                expected_retrievals=["prefer concise code reviews", "onboarding checklist"],
                expected_response_signals=["concise", "onboarding checklist"],
            ),
        ],
    ),
    EvalScenario(
        name="session_recovery",
        description="The agent should recover task context after a persisted session reload.",
        tier="core",
        turns=[
            EvalTurn("Remember that I am debugging a flaky deployment pipeline."),
            EvalTurn("The failure happens during the database migration step."),
            EvalTurn(
                "After reloading this session, what issue was I working on?",
                expected_retrievals=["flaky deployment pipeline", "database migration step"],
                expected_response_signals=["flaky deployment pipeline", "database migration step"],
            ),
        ],
    ),
    EvalScenario(
        name="layered_preferences",
        description="The agent should combine multiple style preferences after unrelated turns.",
        tier="stress",
        turns=[
            EvalTurn("Remember that I prefer concise answers."),
            EvalTurn("Also remember that action items should be in bullet lists."),
            EvalTurn("I am planning a production rollout for a billing service."),
            EvalTurn("Give me three dinner ideas for later tonight."),
            EvalTurn(
                "Now summarize how you should present the rollout plan back to me.",
                expected_retrievals=["prefer concise answers", "bullet lists", "production rollout"],
                expected_response_signals=["concise", "bullet", "rollout plan"],
            ),
        ],
    ),
    EvalScenario(
        name="debugging_resume_with_preference",
        description="The agent should resume a debugging issue while honoring a stored diagnostic preference.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when debugging, I want the root cause hypothesis first."),
            EvalTurn("I am investigating a flaky cache invalidation bug in the user profile service."),
            EvalTurn("The bug appears after deploys when stale profile data persists."),
            EvalTurn("Also remind me to book train tickets tomorrow."),
            EvalTurn(
                "Return to the bug and tell me what to investigate first.",
                expected_retrievals=["root cause hypothesis first", "flaky cache invalidation bug", "stale profile data"],
                expected_response_signals=["root cause", "cache invalidation", "stale profile data"],
            ),
        ],
    ),
    EvalScenario(
        name="coding_agent_resume",
        description="The agent should resume a coding task after an unrelated interruption and honor formatting preferences.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn("I am fixing a flaky retry loop in the worker scheduler."),
            EvalTurn("The suspected problem is duplicate backoff state after task resume."),
            EvalTurn("Also, suggest a lunch option near the office."),
            EvalTurn(
                "Back to the scheduler fix. What should I inspect first?",
                expected_retrievals=["short diff-style summary first", "flaky retry loop", "duplicate backoff state"],
                expected_response_signals=["diff-style", "retry loop", "backoff state"],
            ),
        ],
    ),
    EvalScenario(
        name="artifact_chain_resume",
        description="The agent should recover logs, a debugging hypothesis, and a fix plan after an interruption.",
        tier="stress",
        turns=[
            EvalTurn("Log: worker-7 retry loop overflow after checkpoint resume."),
            EvalTurn("Hypothesis: duplicate scheduler state is rehydrating the same backoff entry twice."),
            EvalTurn("Plan: inspect retry state restore, compare checkpoint IDs, then patch dedupe before requeue."),
            EvalTurn("Also remind me to order coffee beans."),
            EvalTurn(
                "Return to the scheduler incident. What should I inspect first?",
                expected_retrievals=["retry loop overflow", "duplicate scheduler state", "patch dedupe"],
                expected_response_signals=["retry state restore", "checkpoint ids", "dedupe"],
            ),
        ],
    ),
    EvalScenario(
        name="artifact_relation_chain",
        description="The agent should preserve the relation chain from log to hypothesis to plan for the same task.",
        tier="stress",
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect."),
            EvalTurn("Hypothesis: the reconnect path leaves stale cursor state in the sync worker."),
            EvalTurn("Plan: inspect cursor reset, verify reconnect ordering, then patch stale cursor cleanup."),
            EvalTurn(
                "Return to the profile-sync incident and continue the plan.",
                expected_retrievals=["times out after redis reconnect", "stale cursor state", "cursor reset"],
                expected_response_signals=["cursor reset", "reconnect ordering", "stale cursor cleanup"],
            ),
        ],
    ),
    EvalScenario(
        name="blocker_resume_no_reexplaining",
        description="The agent should resume a blocked task without asking the user to restate the blocker.",
        tier="stress",
        turns=[
            EvalTurn("I am blocked on the billing rollout because the migration notes are still incomplete."),
            EvalTurn("The blocker is that rollback steps for the old billing schema are missing."),
            EvalTurn("Also, give me two lunch ideas near the office."),
            EvalTurn(
                "Back to the billing rollout. What is blocking us and what should happen next?",
                expected_retrievals=["migration notes are still incomplete", "rollback steps for the old billing schema are missing"],
                expected_response_signals=["rollback", "migration notes", "next"],
            ),
        ],
    ),
    EvalScenario(
        name="decision_resume_after_interruption",
        description="The agent should preserve a prior decision and continue from it after unrelated interruption.",
        tier="stress",
        turns=[
            EvalTurn("Decision: for the worker scheduler fix, patch dedupe before touching the retry policy."),
            EvalTurn("Reason: duplicate backoff state is the current blocker and we should avoid widening the change."),
            EvalTurn("Also remind me to buy printer paper."),
            EvalTurn(
                "Return to the scheduler work. What decision did we already make and why?",
                expected_retrievals=["patch dedupe before touching the retry policy", "duplicate backoff state is the current blocker"],
                expected_response_signals=["patch dedupe", "duplicate backoff state", "avoid widening the change"],
            ),
        ],
    ),
    EvalScenario(
        name="plan_resume_without_restart",
        description="The agent should continue an existing plan instead of restarting with generic exploration prompts.",
        tier="stress",
        turns=[
            EvalTurn("Plan: inspect retry state restore, compare checkpoint IDs, then patch dedupe before requeue."),
            EvalTurn("The current blocker is duplicate backoff state after checkpoint resume."),
            EvalTurn("I also need dinner ideas for tonight."),
            EvalTurn(
                "Continue the scheduler plan from where we left off.",
                expected_retrievals=["inspect retry state restore", "compare checkpoint IDs", "patch dedupe", "duplicate backoff state"],
                expected_response_signals=["retry state restore", "checkpoint ids", "dedupe", "backoff state"],
            ),
        ],
    ),
    EvalScenario(
        name="contradiction_link_resume",
        description="The agent should resume a task with the strongest remaining root cause instead of resurfacing the ruled-out one.",
        tier="stress",
        tags=["conflict_heavy"],
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect."),
            EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
            EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker."),
            EvalTurn(
                "Return to the profile-sync incident and tell me the strongest remaining root cause.",
                expected_retrievals=["times out after redis reconnect", "stale cursor state remains"],
                expected_response_signals=["stale cursor state", "redis reconnect"],
            ),
        ],
    ),
    EvalScenario(
        name="conflict_unit_dominance",
        description="The agent should preserve the dominant remaining hypothesis and its backing evidence inside a conflict unit.",
        tier="stress",
        tags=["conflict_heavy"],
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and the worker stalls before final checkpoint flush."),
            EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
            EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
            EvalTurn("Hypothesis: the reconnect path is reusing a stale lease token after reconnect."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn(
                "Return to the profile-sync incident and name the strongest remaining root cause.",
                expected_retrievals=["stale cursor state remains", "times out after redis reconnect"],
                expected_response_signals=["stale cursor state", "redis reconnect", "checkpoint flush"],
            ),
        ],
    ),
    EvalScenario(
        name="release_hotfix_handoff",
        description="The agent should resume a release hotfix handoff with the active blocker, chosen scope, and rollback expectation intact.",
        tier="stress",
        turns=[
            EvalTurn("Decision: scope the release hotfix to the scheduler dedupe patch only."),
            EvalTurn("Constraint: avoid widening the change before tonight's rollout window."),
            EvalTurn("The blocker is that rollback notes for the old retry policy are still incomplete."),
            EvalTurn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the rollout checklist."),
            EvalTurn("Also suggest a quick lunch near the office."),
            EvalTurn(
                "Return to the release handoff. What are we shipping, what is still blocking us, and what has to happen next?",
                expected_retrievals=[
                    "scheduler dedupe patch only",
                    "rollback notes for the old retry policy are still incomplete",
                    "verify scheduler dedupe in staging",
                ],
                expected_response_signals=[
                    "scheduler dedupe",
                    "rollback notes",
                    "staging",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="debug_fix_verify_close_loop",
        description="The agent should move from diagnosis to fix to verification without restarting the debugging loop.",
        tier="stress",
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, then verify reconnect ordering in staging before closing the incident."),
            EvalTurn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging."),
            EvalTurn(
                "Before we close this incident, what should we verify and what fix are we carrying forward?",
                expected_retrievals=[
                    "stale cursor cleanup",
                    "reconnect ordering looks stable in staging",
                    "before closing the incident",
                ],
                expected_response_signals=[
                    "stale cursor cleanup",
                    "reconnect ordering",
                    "close",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="coding_review_commitment_chain",
        description="The agent should resume a coding review thread with the active decision, rejected alternative, and next validation step.",
        tier="stress",
        turns=[
            EvalTurn("Decision: keep the retry policy stable and patch dedupe first."),
            EvalTurn("Alternative: changing retry policy first would widen the change too early."),
            EvalTurn("Plan: patch dedupe, rerun scheduler resume tests, then prepare a short diff-style review summary."),
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn(
                "Back to the coding review. What did we commit to, what did we reject, and what should we validate next?",
                expected_retrievals=[
                    "keep the retry policy stable and patch dedupe first",
                    "changing retry policy first would widen the change too early",
                    "rerun scheduler resume tests",
                ],
                expected_response_signals=[
                    "diff-style",
                    "retry policy",
                    "scheduler resume tests",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="incident_root_cause_handoff",
        description="The agent should carry the dominant incident hypothesis and backing evidence into a handoff after interruptions.",
        tier="stress",
        tags=["conflict_heavy", "practical_conflict"],
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
            EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: verify reconnect ordering, collect the redis reconnect evidence, then write the incident handoff."),
            EvalTurn("Also remind me to pay the cloud invoice tomorrow."),
            EvalTurn(
                "Before I hand this incident off, what root cause are we carrying forward and what evidence should I cite?",
                expected_retrievals=[
                    "stale cursor state remains after reconnect",
                    "redis reconnect",
                    "checkpoint flush",
                ],
                expected_response_signals=[
                    "stale cursor state",
                    "redis reconnect",
                    "checkpoint flush",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="release_scope_guardrail_handoff",
        description="The agent should preserve release scope, rejected widening, and the remaining release gate during handoff.",
        tier="stress",
        turns=[
            EvalTurn("Decision: ship only the scheduler dedupe patch in tonight's hotfix."),
            EvalTurn("Alternative: changing the retry policy first would widen the change too early."),
            EvalTurn("Constraint: do not close the rollout until staging verification and rollback notes are complete."),
            EvalTurn("The blocker is that rollback notes for the old retry policy are still incomplete."),
            EvalTurn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the release."),
            EvalTurn("Also, suggest a quick dinner after work."),
            EvalTurn(
                "For the hotfix handoff, what are we shipping, what are we explicitly not doing, and what gate is still open?",
                expected_retrievals=[
                    "scheduler dedupe patch",
                    "changing the retry policy first would widen the change too early",
                    "rollback notes for the old retry policy are still incomplete",
                    "verify scheduler dedupe in staging",
                ],
                expected_response_signals=[
                    "scheduler dedupe",
                    "retry policy",
                    "rollback notes",
                    "staging",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="review_scope_followup_chain",
        description="The agent should preserve what stays in scope, what remains constrained, and what follow-up is deferred in a coding review handoff.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn("Decision: keep the scheduler fix scoped to patch dedupe only."),
            EvalTurn("Constraint: avoid widening the change before tonight's rollout."),
            EvalTurn("Alternative: retry policy rewrite goes into a follow-up change after the hotfix."),
            EvalTurn("Plan: rerun scheduler resume tests, then prepare the review handoff."),
            EvalTurn("Also remind me to buy printer paper."),
            EvalTurn(
                "Back to the review handoff. What did we commit to, what constraint is still active, and what follow-up stays out of scope?",
                expected_retrievals=[
                    "patch dedupe only",
                    "avoid widening the change before tonight's rollout",
                    "retry policy rewrite goes into a follow-up change",
                    "rerun scheduler resume tests",
                ],
                expected_response_signals=[
                    "diff-style",
                    "patch dedupe",
                    "avoid widening the change",
                    "retry policy",
                    "scheduler resume tests",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="procedure_release_handoff_chain",
        description="The agent should preserve the active release handoff checklist as a reusable procedure inside a real handoff chain.",
        tier="stress",
        turns=[
            EvalTurn("Decision: ship only the scheduler dedupe patch in tonight's hotfix."),
            EvalTurn("The blocker is that rollback notes for the old retry policy are still incomplete."),
            EvalTurn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the hotfix."),
            EvalTurn("Release handoff checklist: confirm staging verification, rollback notes, and migration notes before handoff."),
            EvalTurn("Also suggest a quick lunch near the office."),
            EvalTurn(
                "Before handoff, what checklist still applies?",
                expected_retrievals=[
                    "release handoff checklist",
                    "staging verification",
                    "rollback notes",
                    "migration notes",
                ],
                expected_response_signals=[
                    "staging verification",
                    "rollback notes",
                    "migration notes",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="procedure_review_handoff_chain",
        description="The agent should recover the reusable review template during a realistic review-to-handoff chain.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn("Decision: keep the scheduler fix scoped to patch dedupe only."),
            EvalTurn("Plan: rerun scheduler resume tests, then prepare the release review handoff."),
            EvalTurn("Review summary template: start with a diff-style summary, then list risks and follow-up."),
            EvalTurn("Also remind me to buy printer paper."),
            EvalTurn(
                "For the release review handoff, what review format should we use?",
                expected_retrievals=[
                    "review summary template",
                    "diff-style summary",
                    "risks",
                    "follow-up",
                ],
                expected_response_signals=[
                    "diff-style",
                    "risks",
                    "follow-up",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="procedure_release_gate_review_chain",
        description="The agent should recover a release checklist together with the out-of-scope guardrail and final readiness evidence during handoff.",
        tier="stress",
        turns=[
            EvalTurn("Decision: ship only the scheduler dedupe patch in tonight's hotfix."),
            EvalTurn("Alternative: retry policy rewrite stays out of scope until after the hotfix."),
            EvalTurn("Plan: finish migration notes, verify staging verification, and complete rollback notes before handoff."),
            EvalTurn("Release handoff checklist: confirm staging verification, rollback notes, and migration notes before handoff."),
            EvalTurn("Verified: staging verification is complete and rollback notes are now complete for the hotfix handoff."),
            EvalTurn("Also remind me to send the finance update later."),
            EvalTurn("Also suggest a dinner spot near the office."),
            EvalTurn("Also remind me to check the travel booking tomorrow."),
            EvalTurn("Also draft a short note for the infra team."),
            EvalTurn(
                "For the hotfix handoff packet, what checklist still applies, what stays out of scope, and are we ready to hand off?",
                expected_retrievals=[
                    "release handoff checklist",
                    "migration notes",
                    "retry policy rewrite stays out of scope",
                    "staging verification is complete",
                ],
                expected_response_signals=[
                    "migration notes",
                    "retry policy",
                    "ready to hand off",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="procedure_review_validation_handoff_chain",
        description="The agent should recover the review template and the passed validation evidence in the same release review handoff.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn("Decision: keep the scheduler fix scoped to patch dedupe only."),
            EvalTurn("Plan: rerun scheduler resume tests, then prepare the release review handoff."),
            EvalTurn("Review summary template: start with a diff-style summary, then list risks and follow-up."),
            EvalTurn("Verified: scheduler resume tests are green after the dedupe patch."),
            EvalTurn("Also remind me to order coffee filters."),
            EvalTurn("Also suggest a quick lunch near the office."),
            EvalTurn("Also remind me to send the vendor follow-up tomorrow."),
            EvalTurn("Also collect the meeting notes from this morning."),
            EvalTurn(
                "For the release review handoff, what format should the summary use and what validation already passed?",
                expected_retrievals=[
                    "review summary template",
                    "diff-style summary",
                    "follow-up",
                    "scheduler resume tests are green",
                ],
                expected_response_signals=[
                    "diff-style",
                    "follow-up",
                    "scheduler resume tests",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="hotfix_full_lifecycle_replay",
        description="The agent should carry a hotfix from scope decision through blocker tracking, validation, and handoff without restarting.",
        tier="stress",
        turns=[
            EvalTurn("Decision: ship only the scheduler dedupe patch in tonight's hotfix."),
            EvalTurn("Alternative: changing the retry policy first would widen the change too early."),
            EvalTurn("Constraint: keep rollback notes complete before handoff."),
            EvalTurn("The blocker is that rollback notes for the old retry policy are still incomplete."),
            EvalTurn("Plan: verify scheduler dedupe in staging, finish rollback notes, then hand off the hotfix."),
            EvalTurn("Verified: scheduler dedupe looks stable in staging and rollback notes are now complete."),
            EvalTurn("Also remind me to reply to the finance thread later."),
            EvalTurn(
                "Summarize the hotfix handoff: what did we ship, what did we reject, and are we ready to hand off?",
                expected_retrievals=[
                    "scheduler dedupe patch",
                    "changing the retry policy first would widen the change too early",
                    "rollback notes are now complete",
                    "stable in staging",
                ],
                expected_response_signals=[
                    "scheduler dedupe",
                    "retry policy",
                    "staging",
                    "ready to hand off",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="procedure_incident_closeout_replay",
        description="The agent should preserve the incident closeout checklist through a longer debug-fix-verify-close chain.",
        tier="stress",
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff."),
            EvalTurn("Incident closeout checklist: verify reconnect ordering, confirm staging stability, then mark ready to close."),
            EvalTurn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging."),
            EvalTurn("Also suggest a train route for tomorrow morning."),
            EvalTurn("Also remind me to reply to the finance thread later."),
            EvalTurn("Also suggest a quick dinner after work."),
            EvalTurn(
                "Before we close this incident, what checklist still applies?",
                expected_retrievals=[
                    "incident closeout checklist",
                    "verify reconnect ordering",
                    "staging stability",
                    "ready to close",
                ],
                expected_response_signals=[
                    "reconnect ordering",
                    "staging stability",
                    "ready to close",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="procedure_incident_handoff_closeout_replay",
        description="The agent should carry procedure, root-cause evidence, and closeout readiness through a longer incident handoff-to-close chain.",
        tier="stress",
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff."),
            EvalTurn("Incident closeout checklist: verify reconnect ordering, confirm staging stability, then mark ready to close."),
            EvalTurn("Verified: reconnect ordering looks stable in staging and the stale cursor cleanup patch reproduces cleanly."),
            EvalTurn("Decision: do not close the incident until the handoff notes cite the reconnect evidence."),
            EvalTurn("Also suggest a quick dinner after work."),
            EvalTurn("Also remind me to reply to the finance thread later."),
            EvalTurn(
                "Before we close this incident after handoff, what checklist still applies, what evidence is complete, and are we ready to close?",
                expected_retrievals=[
                    "incident closeout checklist",
                    "stale cursor cleanup patch reproduces cleanly",
                    "reconnect ordering looks stable in staging",
                    "ready to close",
                ],
                expected_response_signals=[
                    "reconnect ordering",
                    "stale cursor cleanup",
                    "ready to close",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="sidecar_incident_close_packet_paraphrase",
        description="Sidecar: the agent should preserve closeout readiness when the user asks with weaker close-packet paraphrases.",
        tier="stress",
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff."),
            EvalTurn("Incident closeout checklist: verify reconnect ordering, confirm staging stability, then mark ready to close."),
            EvalTurn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging."),
            EvalTurn("Also remind me to send the vendor follow-up tomorrow."),
            EvalTurn("Also remind me to pay the cloud invoice next week."),
            EvalTurn("Also suggest a quick lunch near the station."),
            EvalTurn(
                "For the close packet, which fix survives, which staging signal is our keeper, and is closure unlocked?",
                expected_retrievals=[
                    "stale cursor cleanup patch reproduces cleanly",
                    "reconnect ordering looks stable in staging",
                    "incident closeout checklist",
                    "ready to close",
                ],
                expected_response_signals=[
                    "stale cursor cleanup",
                    "staging",
                    "ready to close",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="sidecar_incident_handoff_bundle_paraphrase",
        description="Sidecar: the agent should preserve root cause, fix, and citeable evidence under a handoff-bundle paraphrase.",
        tier="stress",
        tags=["conflict_heavy", "practical_conflict"],
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
            EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff."),
            EvalTurn("Incident handoff bundle: carry forward the surviving explanation, surviving patch, reconnect evidence, and checkpoint-flush evidence."),
            EvalTurn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging."),
            EvalTurn("Also suggest a quick train route for tomorrow morning."),
            EvalTurn("Also remind me to send the monthly infra update."),
            EvalTurn("Also suggest a coffee spot near the office."),
            EvalTurn(
                "For the handoff bundle, which explanation survived, which patch survived, and what evidence stays citeable?",
                expected_retrievals=[
                    "stale cursor state remains after reconnect",
                    "stale cursor cleanup patch reproduces cleanly",
                    "reconnect ordering looks stable in staging",
                    "checkpoint flush",
                ],
                expected_response_signals=[
                    "stale cursor state",
                    "stale cursor cleanup",
                    "reconnect ordering",
                    "checkpoint flush",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="incident_conflict_packet_resolution",
        description="Conflict practical: the agent should preserve the surviving theory, the ruled-out theory, and the citeable proof inside a conflict-heavy incident packet.",
        tier="stress",
        tags=["conflict_heavy", "practical_conflict"],
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
            EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident packet."),
            EvalTurn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging."),
            EvalTurn("Also remind me to send the vendor update tomorrow morning."),
            EvalTurn("Also suggest a quick breakfast near the station."),
            EvalTurn("Also remind me to pay the cloud invoice later this week."),
            EvalTurn(
                "For the incident packet, which theory still survives, which theory stays ruled out, and which proof points still travel with it?",
                expected_retrievals=[
                    "stale cursor state remains after reconnect",
                    "cursor reset timing is causing the timeout",
                    "stale cursor cleanup patch reproduces cleanly",
                    "reconnect ordering looks stable in staging",
                    "checkpoint flush",
                ],
                expected_response_signals=[
                    "stale cursor state",
                    "cursor reset timing",
                    "stale cursor cleanup",
                    "checkpoint flush",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="sidecar_release_packet_followthrough_paraphrase",
        description="Sidecar: the agent should preserve release packet scope, follow-up boundary, and validation under packet/ship paraphrases.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn("Decision: keep the scheduler fix scoped to patch dedupe only."),
            EvalTurn("Constraint: avoid widening the change before tonight's rollout."),
            EvalTurn("Alternative: retry policy rewrite goes into a follow-up patch after the hotfix."),
            EvalTurn("Plan: patch dedupe, rerun scheduler resume tests, then prepare the release review handoff."),
            EvalTurn("Release packet procedure: start with a diff-style summary, then list committed scope, deferred retry policy follow-up, validated checks, avoid widening the change, and the ship call."),
            EvalTurn("Verified: scheduler resume tests are green after the dedupe patch."),
            EvalTurn("Also remind me to order coffee filters."),
            EvalTurn("Also remind me to send the vendor follow-up tomorrow."),
            EvalTurn("Also suggest a quick dinner after work."),
            EvalTurn(
                "For tonight's packet, which patch stays in, which follow-up stays out, what validation is already banked, and is the packet clear to ship?",
                expected_retrievals=[
                    "patch dedupe only",
                    "retry policy rewrite goes into a follow-up patch",
                    "scheduler resume tests are green",
                    "avoid widening the change before tonight's rollout",
                    "diff-style summary",
                ],
                expected_response_signals=[
                    "diff-style",
                    "patch dedupe",
                    "retry policy",
                    "scheduler resume tests",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="robustness_incident_story_bundle",
        description="Robustness sidecar: the agent should recover the surviving incident story without relying on explicit handoff/closeout trigger words.",
        tier="stress",
        tags=["conflict_heavy"],
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
            EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff."),
            EvalTurn("Incident handoff bundle: carry forward the surviving explanation, surviving patch, reconnect evidence, and checkpoint-flush evidence."),
            EvalTurn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging."),
            EvalTurn("Also suggest a quick coffee spot near the office."),
            EvalTurn("Also remind me to send the infra update tomorrow."),
            EvalTurn(
                "What story still holds, what fix still holds, and which proof points are still worth citing?",
                expected_retrievals=[
                    "stale cursor state remains after reconnect",
                    "stale cursor cleanup patch reproduces cleanly",
                    "reconnect ordering looks stable in staging",
                    "checkpoint flush",
                ],
                expected_response_signals=[
                    "stale cursor state",
                    "stale cursor cleanup",
                    "reconnect ordering",
                    "checkpoint flush",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="robustness_release_scope_bundle",
        description="Robustness sidecar: the agent should recover release scope and deferred follow-up under de-triggered bundle phrasing.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn("Decision: keep the scheduler fix scoped to patch dedupe only."),
            EvalTurn("Constraint: avoid widening the change before tonight's rollout."),
            EvalTurn("Alternative: retry policy rewrite goes into a follow-up patch after the hotfix."),
            EvalTurn("Plan: patch dedupe, rerun scheduler resume tests, then prepare the release review handoff."),
            EvalTurn("Release packet procedure: start with a diff-style summary, then list committed scope, deferred retry policy follow-up, validated checks, avoid widening the change, and the ship call."),
            EvalTurn("Verified: scheduler resume tests are green after the dedupe patch."),
            EvalTurn("Also remind me to order coffee filters."),
            EvalTurn("Also suggest a quick dinner after work."),
            EvalTurn(
                "What still belongs in tonight's release story, what is still deferred, and what validation already holds?",
                expected_retrievals=[
                    "patch dedupe only",
                    "retry policy rewrite goes into a follow-up patch",
                    "scheduler resume tests are green",
                    "avoid widening the change before tonight's rollout",
                    "diff-style summary",
                ],
                expected_response_signals=[
                    "diff-style",
                    "patch dedupe",
                    "retry policy",
                    "scheduler resume tests",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="incident_debug_handoff_replay",
        description="The agent should preserve the strongest remaining incident explanation, fix direction, and closeout evidence across a long interruption.",
        tier="stress",
        turns=[
            EvalTurn("Log: profile-sync job times out after redis reconnect and stalls before checkpoint flush."),
            EvalTurn("Hypothesis: cursor reset timing is causing the timeout in the sync worker."),
            EvalTurn("The cursor reset timing theory was wrong and ruled out after reproducing the timeout."),
            EvalTurn("Hypothesis: stale cursor state remains after reconnect and blocks the sync worker before checkpoint flush."),
            EvalTurn("Plan: patch stale cursor cleanup, verify reconnect ordering in staging, then prepare the incident handoff."),
            EvalTurn("Verified: stale cursor cleanup patch reproduces cleanly and reconnect ordering looks stable in staging."),
            EvalTurn("Also suggest a train route for tomorrow morning."),
            EvalTurn(
                "Before handoff, what root cause are we carrying forward, what fix are we carrying forward, and what evidence should I cite?",
                expected_retrievals=[
                    "stale cursor state remains after reconnect",
                    "stale cursor cleanup",
                    "reconnect ordering looks stable in staging",
                    "checkpoint flush",
                ],
                expected_response_signals=[
                    "stale cursor state",
                    "stale cursor cleanup",
                    "reconnect ordering",
                    "checkpoint flush",
                ],
            ),
        ],
    ),
    EvalScenario(
        name="review_to_release_followthrough_replay",
        description="The agent should preserve coding-review commitments, validation work, and release follow-through across a longer chain.",
        tier="stress",
        turns=[
            EvalTurn("Remember that when giving code changes, I want a short diff-style summary first."),
            EvalTurn("Decision: keep the scheduler fix scoped to patch dedupe only."),
            EvalTurn("Constraint: avoid widening the change before tonight's rollout."),
            EvalTurn("Alternative: retry policy rewrite goes into a follow-up patch after the hotfix."),
            EvalTurn("Plan: patch dedupe, rerun scheduler resume tests, then prepare the release review handoff."),
            EvalTurn("Verified: scheduler resume tests are green after the dedupe patch."),
            EvalTurn("Also remind me to order coffee filters."),
            EvalTurn(
                "For the release review handoff, what did we commit to, what stays out of scope, and what validation already passed?",
                expected_retrievals=[
                    "patch dedupe only",
                    "retry policy rewrite goes into a follow-up patch",
                    "avoid widening the change before tonight's rollout",
                    "scheduler resume tests are green",
                ],
                expected_response_signals=[
                    "diff-style",
                    "patch dedupe",
                    "retry policy",
                    "scheduler resume tests",
                ],
            ),
        ],
    ),
]
