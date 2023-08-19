"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import bisect
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import scipy.stats as st
from pydantic import confloat, validate_arguments
from scipy.optimize import bisect as scipy_bisect
from scipy.special import ndtr


class GroupSequentialTest:
    def __init__(self) -> None:
        """Instantiate group sequential class."""

    @validate_arguments
    def get_group_sequential_result(
        self,
        point_estimate: float,
        standard_error: float,
        information_rates: List[confloat(gt=0, le=1.0)],
        target_sample_size: int,
        current_sample_size: int,
        **kwargs,
    ) -> Tuple[float, float, float]:
        """Run group sequential testing and adjust confidence intervals and p-values.

        This method is primarily designed to be called within causal-platform pipelines.

        Note that we round the target sample size and information rates to ensure convergence.

        Args:
            point_estimate (float): effect size estimate.
            standard_error (float): standard error of the effect size.
            information_rates (List[confloat, optional): A monotonically increasing ratio of sample size information rates. (e.g., [0.1, 0.5, 1.0], which corresponds to peeking at 10%, 50%, and 100% of the target sample size).
            target_sample_size (int): the required target sample size (this is the total sample and not for each individual group)
            current_sample_size (int): the observed sample size. This is used to identify the most proximate intermediate analysis.

        Returns:
            Tuple(float, float. float): Return the p_value and the left and the right confidence intervals.
        """
        precision = 3
        ratio_so_far = round(min(current_sample_size / target_sample_size, 1.0), precision)
        information_rates = sorted(list(set([round(i, precision) for i in information_rates])))
        gamma = kwargs.get("gamma", -6)
        alpha = kwargs.get("alpha", 0.05)
        sided = kwargs.get("sided", 2)
        # plug in ratio_so_far
        if ratio_so_far not in information_rates:
            bisect.insort(information_rates, ratio_so_far)

        # make sure we always end up with 1.0 at the endpoint.
        information_rates[-1] = min(information_rates[-1], 1.0)

        index = information_rates.index(ratio_so_far)

        # generate p-value
        observed_z = np.abs(point_estimate / standard_error)
        p_value = self.repeated_p_value(
            period_index=index,
            observed_z=observed_z,
            information_rates=np.array(information_rates),
            gamma=gamma,
        )
        if sided == 2:
            p_value = min(1.0, p_value * 2)
        z_score_critical = self.compute_ci_boundry(
            tuple(information_rates), alpha=alpha, gamma=gamma, period_index=index
        )
        ci_interval = standard_error * z_score_critical
        return p_value, point_estimate - ci_interval, point_estimate + ci_interval

    def repeated_p_value(
        self,
        period_index: int,
        observed_z: float,
        information_rates: List[float],
        gamma: float = -6,
        prec: float = 0.001,
    ) -> float:
        """Calculate the p-value for a repeated hypothesis test.

        Args:
            period_index (int): The index of the period for which to calculate the p-value.
            observed_z (float): The observed z-score.
            information_rates (List[float]): The information rates used in sequential testing boundary generation.
            gamma (float, optional): The spending function parameter used by Hwan-Shih-Decani. Defaults to -6.
            prec (float, optional): The tolerance for the bisection method. Defaults to 0.001.

        Returns:
            float: The calculated p-value of the observed_z in the context of sequential testing.
        """

        def _fun(x: float) -> float:
            """Specify the objective function for the bisection method.

            Args:
                x (float): The p-value.

            Returns:
                float: The difference between the observed z-score and the boundary.
            """
            boundary = self.comp_bounds(information_rates=information_rates, alpha=x, gamma=gamma, ztrun=8, prec=prec)[
                period_index
            ]
            return observed_z - boundary

        return self.bisearch(_fun, [0.00000001, 0.99999999], tol=prec)

    @lru_cache(typed=True)
    def compute_ci_boundry(
        self, information_rates: Tuple[float], alpha: float, gamma: float, period_index: int
    ) -> float:
        """Get the z score that is used to compute the confidence interval.

        Args:
            information_rates (Tuple[float]): Information rates for which we need to compute boundaries.
            alpha (float): alpha level.
            gamma (float): gamma parameter for the spending function.
            period_index (int): the current observation index. If you peek 10 times and index is 2, we simply return the z-score at that index.

        Returns:
            float: Boundary z score at an observation index.
        """
        critical_boundries = self.comp_bounds(information_rates=np.array(information_rates), alpha=alpha, gamma=gamma)
        return critical_boundries[period_index]

    def bisearch(self, f, interval, tol=2**-2, maxiter=1000):
        """Find the root of a function using bisection method.

        Args:
            f (function): A function to find the root of.
            interval (tuple): A tuple of two numbers representing the interval in which to search for a root.
            tol (float, optional): The tolerance for the solution. Defaults to 2**-2.
            maxiter (int, optional): The maximum number of iterations to perform. Defaults to 1000.

        Returns:
            float: The root of the function `f` within the given interval.

        Raises:
            Exception: If the values of `f` at the interval's endpoints have the same sign.
        """
        lower, upper = min(interval), max(interval)

        for i in range(maxiter):
            mid = (lower + upper) / 2
            mid_result = f(mid)
            if mid_result > 0:
                upper = mid
            else:
                lower = mid
            if abs(lower - upper) <= tol:
                break

        if i == maxiter - 1:
            print("maximum number of iterations reached, precision may be insufficient")

        root = (lower + upper) / 2
        return root

    def cumulative_probability_of_failure(
        self,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        information_rates: np.ndarray,
        intervals: np.ndarray,
    ) -> np.ndarray:
        """Calculate cumulative probability of rejecting the null given peeking cadence.

        Args:
            lower_bound (np.ndarray): Lower bound of each stage.
            upper_bound (np.ndarray): Upper bound of each stage.
            information_times (np.ndarray): Information times of each stage.
            intervals (np.ndarray): Intervals of discretization.

        Returns:
            np.ndarray: The cumulative probability of failure.
        """
        intervals = intervals.astype(int)
        interval_step = (upper_bound - lower_bound) / intervals
        num_stages = len(lower_bound)
        cumulative_prob_upper = np.ones((num_stages, 1))
        sqrt_2pi = np.sqrt(2 * np.pi)
        uniform_grid = np.arange(1, intervals[0] + 1)
        E = np.ones((1, intervals[0]))

        discretized_x = lower_bound[0] + ((uniform_grid) - 0.5 * E) * interval_step[0]

        cumulative_prob_upper[0] = ndtr(
            -(np.sqrt(information_rates[0]) * upper_bound[0]) / np.sqrt(information_rates[0])
        )

        intermediate_result = np.transpose(
            (interval_step[0] / sqrt_2pi)
            * np.exp(-((np.sqrt(information_rates[0]) * discretized_x) ** 2) / (2 * information_rates[0]))
        )

        for stage in range(1, num_stages):
            upper_prob = ndtr(
                -(
                    np.sqrt(information_rates[stage]) * upper_bound[stage] * E
                    - np.sqrt(information_rates[stage - 1]) * discretized_x
                )
                / np.sqrt(information_rates[stage] - information_rates[stage - 1])
            )

            cumulative_prob_upper[stage] = cumulative_prob_upper[stage - 1] + np.dot(upper_prob, intermediate_result)
            x = lower_bound[stage] + ((np.arange(1, intervals[stage] + 1)) - 0.5) * interval_step[stage]
            if stage != num_stages - 1:
                A = (
                    interval_step[stage]
                    * np.sqrt(information_rates[stage])
                    / (sqrt_2pi * np.sqrt(information_rates[stage] - information_rates[stage - 1]))
                )

                B = np.exp(
                    -(
                        np.square(
                            np.sqrt(information_rates[stage]) * x.reshape(-1, 1)
                            - np.sqrt(information_rates[stage - 1]) * discretized_x
                        )
                    )
                    / (2 * (information_rates[stage] - information_rates[stage - 1]))
                )

                intermediate_result = np.matmul(A * B, intermediate_result)
                discretized_x = x
        return cumulative_prob_upper

    def comp_bounds(
        self,
        information_rates: np.array,
        alpha: float = 0.025,
        gamma: float = -6,
        ztrun: float = 8,
        prec: float = 0.001,
    ) -> np.ndarray:
        """Compute z-score bounds for a sequence of information rates.

        Args:
            information_rates (numpy array): An array of information rates.
            alpha (float): The significance level under which we test rejecting the null.
            gamma (float): A parameter related to the spending function we use.
            ztrun (float): The truncation value. Values with z-scores above 6 are truncated.
            prec (float): Minimum precision tolerance when finding the uniroot.

        Returns:
            numpy array: An array of z-scores for every interim analysis. If observed z-score > the boundry, we can reject the null.

        Raises:
            ValueError: If `gamma` is equal to 0.
        """
        n = len(information_rates)
        n_intervals = 75

        if gamma == 0:
            raise ValueError("gamma must be unequal 0")

        if n > 1:
            levsp = alpha * (1 - np.exp(-gamma * information_rates / information_rates[-1])) / (1 - np.exp(-gamma))
            bounds = np.zeros(n)
            bounds[0] = st.norm.ppf(1 - levsp[0])

            for i in range(1, n):
                bmin = st.norm.ppf(1 - levsp[i])
                bmax = st.norm.ppf(1 - levsp[i] + levsp[i - 1])

                if abs(bmin - bmax) <= prec:
                    bounds[i] = bmin
                else:
                    bounds[i] = scipy_bisect(
                        lambda x: self.cumulative_probability_of_failure(
                            lower_bound=np.repeat(-ztrun, i + 1),
                            upper_bound=np.concatenate([bounds[:i], [x]]),
                            information_rates=information_rates[: i + 1],
                            intervals=n_intervals * np.ones(i + 1),
                        )[-1]
                        - levsp[i],
                        bmin,
                        bmax,
                        xtol=prec,
                    )

            return bounds

        return st.norm.ppf(1 - alpha)
