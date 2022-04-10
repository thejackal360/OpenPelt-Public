import pytest
import numpy as np

from basic_bang_bang import bang_bang
from pid_hot import pid_hot
from pid_cold import pid_cold
from transient import transient


def test_bang_bang_controller(ref_temp=65.00):
    (current, heat, cool), _ = bang_bang(reference_temperatures=[ref_temp])
    ref_vals = np.ones(heat[1, 200:].shape) * ref_temp
    assert np.allclose(heat[1, 200:], ref_vals, rtol=1, atol=1) is True


def test_pid_hot(ref_temp=75.00):
    (voltage, heat, cool), _ = pid_hot(ref_temperatures=[ref_temp])
    ref_vals = np.ones(heat[1, 200:].shape) * ref_temp
    assert np.allclose(heat[1, 200:], ref_vals, rtol=1, atol=1) is True


def test_pid_cold(ref_temp=35.00):
    (voltage, heat, cool), _ = pid_cold(ref_temperatures=[ref_temp])
    ref_vals = np.ones(heat[1, 200:].shape) * ref_temp
    assert np.allclose(heat[1, 200:], ref_vals, rtol=1, atol=1) is True


def test_transient():
    (current, heat, cool), _ = transient()
    assert current[1, -1] - 2.1 < 1e-3
    assert np.abs(heat[1, -1] - 29.0) < 1.0
    assert np.abs(cool[1, -1] - 1.0) < 1.0
