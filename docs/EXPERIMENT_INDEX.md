# Experiment Index

This index tracks the first batch of research scripts moved out of `src/` into `experiments/`.

## Verification Experiments

- `src/Q1_lambda_c_scaling.py` -> `experiments/verification/Q1_lambda_c_scaling.py`
- `src/Q1_N_att_scaling.py` -> `experiments/verification/Q1_N_att_scaling.py`
- `src/Q2_lambda_mu_phase.py` -> `experiments/verification/Q2_lambda_mu_phase.py`
- `src/Q3_noise_escape.py` -> `experiments/verification/Q3_noise_escape.py`
- `src/Q4_asymmetric_k.py` -> `experiments/verification/Q4_asymmetric_k.py`
- `src/verify_layered_cascade.py` -> `experiments/verification/verify_layered_cascade.py`
- `src/verification_a.py` -> `experiments/verification/verification_a.py`
- `src/verification_a_v2.py` -> `experiments/verification/verification_a_v2.py`
- `src/verification_b.py` -> `experiments/verification/verification_b.py`

## Control Experiments

- `src/verify_control_strategies.py` -> `experiments/control/verify_control_strategies.py`
- `src/verify_ppo_control.py` -> `experiments/control/verify_ppo_control.py`
- `src/experiment_drift_baseline.py` -> `experiments/control/experiment_drift_baseline.py`
- `src/experiment_fusion_block.py` -> `experiments/control/experiment_fusion_block.py`
- `src/experiment_k1_L1_baseline.py` -> `experiments/control/experiment_k1_L1_baseline.py`
- `src/experiment_rule_perturbation.py` -> `experiments/control/experiment_rule_perturbation.py`

## Compatibility Policy

Legacy wrapper files remain in `src/` for now. New work should import or execute the scripts from `experiments/`.
