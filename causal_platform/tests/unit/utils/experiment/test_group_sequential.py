"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
import pytest

from causal_platform.src.utils.experiment.group_sequential import GroupSequentialTest

group = GroupSequentialTest()


@pytest.mark.parametrize(
    "information_rates,expected_critical_values",
    [
        (
            np.linspace(0.1, 1.0, 10),
            [
                3.885440,
                3.719515,
                3.538037,
                3.348987,
                3.154156,
                2.950575,
                2.737329,
                2.511047,
                2.271144,
                2.012370,
            ],
        ),
        (np.linspace(0.2, 1.0, 5), [3.625624, 3.284455, 2.902256, 2.475946, 1.988669]),
        (np.array([0.5, 1.0]), [3.039299, 1.966126]),
        (
            np.array([0.001, 0.002, 0.004, 0.5, 1.0]),
            [4.948542, 4.940778, 4.803394, 3.039681, 1.966140],
        ),
    ],
)
def test_that_boundary_gets_generated_correctly(information_rates, expected_critical_values):
    result_pvalue = group.comp_bounds(
        information_rates=information_rates,
        alpha=0.025,
        gamma=-6,
        prec=0.0001,
    )
    assert result_pvalue == pytest.approx(
        expected_critical_values, abs=0.001
    ), "Group sequential test should generate correct z-scores"


@pytest.mark.parametrize(
    "test_zscore,expected_pvalue",
    [
        (0.1, 0.9999695),
        (1.0, 0.5476989),
        (1.5, 0.2365417),
        (2.5, 0.02340698),
        (3.5, 0.0009460449),
        (4.5, 0.00003051758),
    ],
)
def test_repeated_p_value_generation(test_zscore, expected_pvalue):
    result_pvalue = group.repeated_p_value(
        period_index=3,
        observed_z=test_zscore,
        information_rates=np.linspace(0.2, 1.0, 5),
        gamma=-6,
        prec=0.0001,
    )
    assert result_pvalue == pytest.approx(
        expected_pvalue, abs=0.001
    ), "Group sequential test should generate correct pvalue"


def test_that_ci_and_p_values_are_correct():
    p_value, lower, upper = group.get_group_sequential_result(
        0,
        1,
        [0.2, 0.4, 0.6, 0.8, 1.0],
        current_sample_size=20,
        target_sample_size=100,
        alpha=0.025,
    )
    assert p_value == pytest.approx(1.0, 0.001)
    assert lower == pytest.approx(-3.625, 0.001)
    assert upper == pytest.approx(3.625, 0.001)

    p_value, lower, upper = group.get_group_sequential_result(
        3.625,
        1,
        [0.2, 0.4, 0.6, 0.8, 1.0],
        current_sample_size=20,
        target_sample_size=100,
        alpha=0.025,
    )
    assert p_value == pytest.approx(0.0498, 0.001)
    assert lower == pytest.approx(-0.0006242, 0.0001)
    assert upper == pytest.approx(7.2506, 0.001)


def test_for_numerical_precision():
    """Checks that we don't raise f(a) and f(b) must have different signs"""
    information_rates = [
        0.001,
        0.002,
        0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.069,
        0.103,
        0.138,
        0.172,
        0.207,
        0.241,
        0.483,
        0.724,
        0.966,
        1,
    ]
    p_value, lower, upper = group.get_group_sequential_result(
        0, 1, information_rates, target_sample_size=28, current_sample_size=27
    )
    assert p_value is not None


def test_that_sequential_output_converges():
    """
    Check that we converge when differences  between information rates are <0.01%
    """
    p_value, _, _ = group.get_group_sequential_result(
        0,
        1,
        [0.2, 0.2001, 0.4, 0.6, 0.8, 1.0],
        current_sample_size=400001,
        target_sample_size=1000000,
        alpha=0.025,
    )
    assert p_value
