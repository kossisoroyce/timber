"""Tests for WCET (Worst-Case Execution Time) analysis."""

import pytest

from timber.accel.safety.realtime.wcet import analyze_wcet, ARCH_COSTS


class TestWCETTreeEnsemble:
    def test_simple_ensemble_cycles_positive(self, simple_ensemble):
        result = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=100.0)
        assert result["total_cycles_worst"] > 0
        assert result["total_cycles_avg"] > 0

    def test_simple_ensemble_time_positive(self, simple_ensemble):
        result = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=100.0)
        assert result["total_time_us_worst"] > 0.0
        assert result["total_time_us_avg"] > 0.0

    def test_worst_case_ge_average(self, simple_ensemble):
        result = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=100.0)
        assert result["total_cycles_worst"] >= result["total_cycles_avg"]

    def test_per_stage_breakdown(self, simple_ensemble):
        result = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=100.0)
        assert len(result["per_stage"]) == 1
        stage = result["per_stage"][0]
        assert stage["stage"] == "trees"
        assert stage["stage_type"] == "tree_ensemble"
        assert stage["cycles_worst"] > 0

    def test_multi_tree_higher_than_single(self, simple_ensemble, multi_tree_ensemble):
        single = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=100.0)
        multi = analyze_wcet(multi_tree_ensemble, arch="cortex-m4", clock_mhz=100.0)
        assert multi["total_cycles_worst"] > single["total_cycles_worst"]


class TestWCETLinear:
    def test_linear_model_cycles_positive(self, linear_ir):
        result = analyze_wcet(linear_ir, arch="cortex-m4", clock_mhz=100.0)
        assert result["total_cycles_worst"] > 0

    def test_linear_per_stage(self, linear_ir):
        result = analyze_wcet(linear_ir, arch="cortex-m4", clock_mhz=100.0)
        assert len(result["per_stage"]) == 1
        assert result["per_stage"][0]["stage_type"] == "linear"


class TestWCETArchitectures:
    @pytest.mark.parametrize("arch", list(ARCH_COSTS.keys()))
    def test_supported_architectures(self, simple_ensemble, arch):
        result = analyze_wcet(simple_ensemble, arch=arch, clock_mhz=100.0)
        assert result["arch"] == arch
        assert result["total_cycles_worst"] > 0

    def test_unknown_arch_raises(self, simple_ensemble):
        with pytest.raises(ValueError, match="Unknown architecture"):
            analyze_wcet(simple_ensemble, arch="nonexistent_arch")


class TestWCETClockFrequency:
    def test_higher_clock_lower_time(self, simple_ensemble):
        slow = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=50.0)
        fast = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=200.0)
        assert fast["total_time_us_worst"] < slow["total_time_us_worst"]
        # Cycle counts should be the same regardless of clock
        assert fast["total_cycles_worst"] == slow["total_cycles_worst"]

    def test_clock_mhz_recorded(self, simple_ensemble):
        result = analyze_wcet(simple_ensemble, arch="cortex-m4", clock_mhz=168.0)
        assert result["clock_mhz"] == 168.0
