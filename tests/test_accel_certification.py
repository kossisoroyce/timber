"""Tests for certification compliance checker and report generation."""

import pytest


class TestComplianceChecker:
    """Test compliance profile rule checking."""

    def test_clean_code_passes_basic_rules(self):
        from timber.accel.safety.certification.profiles import check_compliance

        clean_code = """\
#include "model.h"
#include <math.h>

static float compute(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}
"""
        result = check_compliance(clean_code, "do_178c")
        assert "rules_checked" in result
        assert result["rules_checked"] > 0
        assert isinstance(result["violations"], list)

    def test_malloc_fails_no_dynamic_allocation(self):
        from timber.accel.safety.certification.profiles import check_compliance

        bad_code = """\
#include <stdlib.h>
float* allocate(int n) {
    float* p = (float*)malloc(n * sizeof(float));
    return p;
}
"""
        result = check_compliance(bad_code, "do_178c")
        has_alloc_violation = any(
            "dynamic" in str(v).lower() or "alloc" in str(v).lower()
            for v in result["violations"]
        )
        assert has_alloc_violation

    def test_recursion_detected(self):
        from timber.accel.safety.certification.profiles import check_compliance

        recursive_code = """\
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""
        result = check_compliance(recursive_code, "do_178c")
        has_recursion = any(
            "recurs" in str(v).lower() for v in result["violations"]
        )
        assert has_recursion


class TestCertificationReport:
    """Test certification report generation."""

    def test_generate_report(self, simple_ensemble):
        from timber.accel.safety.certification.report import generate_certification_report

        report = generate_certification_report(simple_ensemble, "do_178c")
        assert report.standard is not None
        assert report.model_summary is not None
        assert report.compliance_result is not None

    def test_report_json_roundtrip(self, simple_ensemble):
        from timber.accel.safety.certification.report import generate_certification_report
        import json

        report = generate_certification_report(simple_ensemble, "do_178c")
        data = json.loads(report.to_json())
        assert "standard" in data
        assert "compliance_result" in data

    def test_report_with_wcet(self, simple_ensemble):
        from timber.accel.safety.certification.report import generate_certification_report

        report = generate_certification_report(
            simple_ensemble, "do_178c", include_wcet=True
        )
        assert report.wcet_result is not None
        assert report.wcet_result["total_cycles_worst"] > 0

    def test_report_summary_string(self, simple_ensemble):
        from timber.accel.safety.certification.report import generate_certification_report

        report = generate_certification_report(simple_ensemble, "do_178c")
        summary = report.summary()
        assert "Certification Report" in summary
        assert "Rules checked" in summary
