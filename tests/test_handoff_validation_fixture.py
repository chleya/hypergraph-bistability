#!/usr/bin/env python
"""Test for handoff validation fixture and regression reporting."""
import json
import os
from pathlib import Path


def load_fixture(fixture_path: str = None):
    """Load the handoff validation fixture."""
    if fixture_path is None:
        fixture_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "fixtures", 
            "handoff_validation_round1.json"
        )
    
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_fixture_schema():
    """Test that the fixture has the required schema fields."""
    fixture = load_fixture()
    
    # Check top-level fields
    assert "round" in fixture
    assert "description" in fixture
    assert "created_at" in fixture
    assert "validator_version" in fixture
    assert "total_samples" in fixture
    assert "samples" in fixture
    
    # Check first sample has all required fields
    required_fields = [
        "sample_id",
        "scene_type",
        "user_prompt",
        "handoff_bundle",
        "response",
        "validator_status",
        "validator_details",
        "human_label",
        "agreement",
        "error_reason"
    ]
    
    for sample in fixture["samples"]:
        for field in required_fields:
            assert field in sample, f"Missing field {field} in sample {sample.get('sample_id', 'unknown')}"


def test_fixture_status_values():
    """Test that validator_status and human_label use valid values."""
    fixture = load_fixture()
    
    valid_validator_statuses = {"fail", "weak_pass", "strong_pass"}
    valid_human_labels = {"fail", "weak_pass", "strong_pass", "marginal"}
    
    for sample in fixture["samples"]:
        assert sample["validator_status"] in valid_validator_statuses, \
            f"Invalid validator_status: {sample['validator_status']}"
        assert sample["human_label"] in valid_human_labels, \
            f"Invalid human_label: {sample['human_label']}"


def test_fixture_scene_types():
    """Test that scene_type values are valid."""
    fixture = load_fixture()
    
    valid_scene_types = {"chat", "coding", "planning", "writing", "research"}
    
    for sample in fixture["samples"]:
        assert sample["scene_type"] in valid_scene_types, \
            f"Invalid scene_type: {sample['scene_type']}"


def test_validation_regression_report():
    """Generate a regression report comparing validator output to human labels."""
    fixture = load_fixture()
    
    samples = fixture["samples"]
    
    # Count agreements
    true_positive = 0  # validator=fail, human=fail
    false_positive = 0  # validator=fail, human!=fail
    true_negative = 0  # validator!=fail, human!=fail
    false_negative = 0  # validator!=fail, human=fail
    
    weak_agreements = 0
    strong_agreements = 0
    
    for sample in samples:
        v_status = sample["validator_status"]
        h_label = sample["human_label"]
        agreement = sample["agreement"]
        
        if agreement:
            if v_status == "strong_pass" and h_label == "strong_pass":
                strong_agreements += 1
            elif v_status == "weak_pass" and h_label == "weak_pass":
                weak_agreements += 1
            elif v_status == "fail" and h_label == "fail":
                # fail agreements
                pass
            elif h_label == "marginal" and v_status in ["fail", "weak_pass"]:
                # marginal cases
                pass
    
    # Calculate statistics
    total = len(samples)
    agreements = sum(1 for s in samples if s["agreement"])
    partial_agreements = sum(1 for s in samples if s.get("agreement") == "partial")
    disagreements = total - agreements - partial_agreements
    
    print("\n" + "="*60)
    print("HANDOFF VALIDATION REGRESSION REPORT")
    print("="*60)
    print(f"Round: {fixture['round']}")
    print(f"Validator Version: {fixture['validator_version']}")
    print(f"Total Samples: {total}")
    print(f"\nAgreement Statistics:")
    print(f"  - Full Agreement: {agreements}/{total} ({100*agreements/total:.1f}%)")
    print(f"  - Partial Agreement: {partial_agreements}/{total}")
    print(f"  - Disagreement: {disagreements}/{total}")
    
    # Status breakdown
    print(f"\nValidator Status Breakdown:")
    status_counts = {}
    for s in samples:
        status = s["validator_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"  - {status}: {count}")
    
    print(f"\nHuman Label Breakdown:")
    label_counts = {}
    for s in samples:
        label = s["human_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count}")
    
    # Error analysis
    print(f"\nError Analysis:")
    error_reasons = {}
    for s in samples:
        if not s["agreement"]:
            reason = s.get("error_reason", "unknown")
            # Categorize errors
            if "FALSE NEGATIVE" in reason:
                error_reasons["false_negative"] = error_reasons.get("false_negative", 0) + 1
            elif "FALSE POSITIVE" in reason:
                error_reasons["false_positive"] = error_reasons.get("false_positive", 0) + 1
            else:
                error_reasons["other"] = error_reasons.get("other", 0) + 1
    
    for error_type, count in error_reasons.items():
        print(f"  - {error_type}: {count}")
    
    print("="*60)
    
    # Return report data for assertions
    return {
        "total": total,
        "agreements": agreements,
        "partial_agreements": partial_agreements,
        "disagreements": disagreements,
        "agreement_rate": agreements / total if total > 0 else 0
    }


if __name__ == "__main__":
    # Run the report
    test_fixture_schema()
    test_fixture_status_values()
    test_fixture_scene_types()
    report = test_validation_regression_report()
    
    print(f"\nTest passed! Agreement rate: {report['agreement_rate']*100:.1f}%")
