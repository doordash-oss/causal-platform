"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Tuple

import numpy as np
from scipy.stats import norm


def calculate_default_sequential_tau(se: float):
    """
    from https://docs.google.com/document/d/1 we decided that 4*se**2 is a good value for tau**2
    here it's implemented as 2*se in order to maintain scale with the standard error, and is appropriately rescaled when calculating the sequential test p-value
    """
    return 2 * se


def fixed_compute_p_value_and_conf_int(
    point_estimate: float,
    standard_error: float,
    alpha: float,
) -> Tuple[float, float, float]:
    z_stats = point_estimate / standard_error
    p_value = (1 - norm.cdf(abs(z_stats))) * 2
    conf_int_radius = norm.ppf(1 - alpha / 2) * standard_error
    conf_int_left = point_estimate - conf_int_radius
    conf_int_right = point_estimate + conf_int_radius
    return p_value, conf_int_left, conf_int_right


def seq_compute_p_value_and_conf_int(point_estimate: float, standard_error: float, tau: float, alpha: float):
    """
    sequential test implementation docs
    https://docs.google.com/document/d/1agUGY17tskapolR1y6rOl_iAP9kvo3TXjR7QFPBXGbA/
    http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p1517.pdf
    """
    # TODO: RuntimeWarning: invalid value encountered in double_scalars (fix this to be numerically safer)
    T = tau**2
    V = standard_error**2

    p_value = np.exp((-(point_estimate**2) * T / (V * (T + V)) - np.log(V / (V + T))) * 0.5)
    p_value = np.nanmin([p_value, 1])
    radius = np.sqrt((2 * np.log(1 / alpha) - np.log(V / (V + T))) * (V * (V + T) / T))

    radius = np.nan_to_num(radius, nan=0.0)
    conf_int_left = point_estimate - radius
    conf_int_right = point_estimate + radius
    return p_value, conf_int_left, conf_int_right


def get_treatment_effect_standard_error(var_control, var_treatment):
    return np.sqrt(var_treatment + var_control)


def is_invertible(x):
    return x.shape[0] == x.shape[1] and np.linalg.matrix_rank(x) == x.shape[0]
