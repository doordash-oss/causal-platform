"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from scipy.stats import t


def adjust_result_using_t(est, se, n, p, alpha=0.05):
    """
    Based on model output treatment effect and standard error,
    compute p-value and confidence interval by using t-distribution with df_resid = n - p

    :param est: estimated treatment effect from model
    :param se: estimated standard error from model
    :param n: effective number of units, if cluster exists should be number of clusters
    :param p: number of regressors including intercept
    :param alpha: threshold by default 0.05
    :return: p_value, ci_left, ci_right
    """
    t_stats = est / se
    df_resid = max(n - p, 1)
    ci_radius = t.ppf(1 - alpha / 2, df_resid) * se
    p_value = (1 - t.cdf(abs(t_stats), df_resid)) * 2
    ci_left = est - ci_radius
    ci_right = est + ci_radius
    return p_value, ci_left, ci_right
