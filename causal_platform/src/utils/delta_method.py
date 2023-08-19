"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

"""
Delta Method Functions
Core delta method related functions in causal-platform, including var/cov computation and optimization solution
Methodology please refer to: https://www.overleaf.com/read/xxpttdqfmvvm
For any changes, please add or update tests in test_delta_method file
"""

from typing import List

import numpy as np
import pandas as pd
from numpy.linalg import inv

from causal_platform.src.models.configuration_model.base_objects import Covariate, Metric
from causal_platform.src.utils.logger import logger


def get_delta_ratio_covariance(x, y, u, v):
    """
    Calculates covariance of two ratio metrics
        x: numerator of the 1st ratio metric
        y: denominator of the 1st ratio metric
        u: numerator of the 2nd ratio metric
        v: denominator of the 2nd ratio metric
    """
    k = x.shape[0]
    mu_x = x.mean()
    mu_y = y.mean()
    mu_u = u.mean()
    mu_v = v.mean()

    cov_matrix = np.cov([x, y, u, v])
    cov_xu_avg = cov_matrix[0, 2] / k
    cov_xv_avg = cov_matrix[0, 3] / k
    cov_yu_avg = cov_matrix[1, 2] / k
    cov_yv_avg = cov_matrix[1, 3] / k
    return (
        (mu_x * mu_u)
        / (mu_y * mu_v)
        * (
            cov_xu_avg / (mu_x * mu_u)
            + cov_yv_avg / (mu_y * mu_v)
            - cov_xv_avg / (mu_x * mu_v)
            - cov_yu_avg / (mu_y * mu_u)
        )
    )


def get_delta_ratio_variance(n: np.array, d: np.array):
    """
    Get the variance of sum(n)/sum(d) which can be done by adapting the delta ratio covariance formula
    """
    return get_delta_ratio_covariance(n, d, n, d)


def get_delta_ratio_correlation(x, y, u, v):
    """
    Calculates correlation of two ratio quantities using delta method
    Inputs:
        x: numerator of the 1st ratio metric
        y: denominator of the 1st ratio metric
        u: numerator of the 2nd ratio metric
        v: denominator of the 2nd ratio metric
    """
    return get_delta_ratio_covariance(x, y, u, v) / np.sqrt(
        get_delta_ratio_variance(x, y) * get_delta_ratio_variance(u, v)
    )


def calculate_adjusted_ratio_metric_covariate_covariance(
    data: pd.DataFrame, metric: Metric, processed_covariates: List[Covariate], covariate: Covariate
):
    """
    Compute the covariance of the covariates-adjusted metric and the new covariate Un/Vn
    Cov(X/Y - theta_1 * U1/V1 - theta_2 * U2/V2 - ... theta_n-1 * Un-1/Vn-1, Un/Vn)
    where U1/V1, ... Un1-/Vn-1 comes from processed_covariates
    """
    covariance = get_delta_ratio_covariance(
        data[metric.numerator_column.column_name],
        data[metric.denominator_column.column_name],
        data[covariate.numerator_column.column_name],
        data[covariate.denominator_column.column_name],
    )
    for cov in processed_covariates:
        covariance -= cov.coef * get_delta_ratio_covariance(
            data[cov.numerator_column.column_name],
            data[cov.denominator_column.column_name],
            data[covariate.numerator_column.column_name],
            data[covariate.denominator_column.column_name],
        )
    return covariance


def calculate_ratio_covariate_coefficients(
    data: pd.DataFrame, metric: Metric, covariates: List[Covariate]
) -> List[Covariate]:
    """
    Compute the coefficients of each ratio covariate sequentially given metric
    """
    processed_covariates = []
    for covariate in covariates:
        try:
            covariate.coef = calculate_adjusted_ratio_metric_covariate_covariance(
                data, metric, processed_covariates, covariate
            ) / get_delta_ratio_variance(
                data[covariate.numerator_column.column_name],
                data[covariate.denominator_column.column_name],
            )
            if np.isfinite(covariate.coef):
                processed_covariates.append(covariate)
            else:
                logger.warning(
                    f"Skipping variance reduction on metric {metric.column_name}"
                    f"using covariate {covariate.column_name} as there is no variance in covariate"
                )
        except Exception as e:
            logger.error(
                f"Error while processing covariate {covariate.column_name}"
                f"for metric {metric.column_name} due to {e}"
            )
    return processed_covariates


def calculate_ratio_covariate_coefficients_noniterative(
    data: pd.DataFrame, metric: Metric, covariates: List[Covariate]
) -> List[Covariate]:
    """
    Compute the coefficients of each ratio covariate simultaneously given metric
    """
    n_covs = len(covariates)
    opt_coef_first_term_mat = np.zeros([n_covs, n_covs])
    opt_coef_second_term_vect = np.zeros(n_covs)

    for ith_cov in range(n_covs):
        for jth_cov in range(n_covs):
            opt_coef_first_term_mat[ith_cov][jth_cov] = get_delta_ratio_covariance(
                data[covariates[ith_cov].numerator_column.column_name],
                data[covariates[ith_cov].denominator_column.column_name],
                data[covariates[jth_cov].numerator_column.column_name],
                data[covariates[jth_cov].denominator_column.column_name],
            )

    for ith_cov in range(n_covs):
        opt_coef_second_term_vect[ith_cov] = get_delta_ratio_covariance(
            data[metric.numerator_column.column_name],
            data[metric.denominator_column.column_name],
            data[covariates[ith_cov].numerator_column.column_name],
            data[covariates[ith_cov].denominator_column.column_name],
        )

    opt_coefs = np.matmul(inv(opt_coef_first_term_mat), opt_coef_second_term_vect)

    processed_covariates = []
    for ith_cov in range(n_covs):
        covariates[ith_cov].coef = opt_coefs[ith_cov]
        if np.isfinite(covariates[ith_cov].coef):
            processed_covariates.append(covariates[ith_cov])
        else:
            logger.warning(
                f"Skipping variance reduction on metric {metric.column_name}"
                f"using covariate {covariates[ith_cov].column_name} as there is no variance in covariate"
            )
    return processed_covariates, opt_coef_first_term_mat


def get_adjusted_ratio_metric_variance(data, metric: Metric, processed_covariates: List[Covariate]):
    """
    Compute the variance of the covariates-adjusted metric
    Var(X/Y - theta_1 * U1/V1 - theta_2 * U2/V2 - ... theta_n * Un/Vn)
    where U1/V1, ... Un/Vn comes from processed_covariates
    """
    variance = get_delta_ratio_variance(
        data[metric.numerator_column.column_name],
        data[metric.denominator_column.column_name],
    )
    if len(processed_covariates) > 0:
        for covariate in processed_covariates:
            variance -= (
                2
                * covariate.coef
                * get_delta_ratio_covariance(
                    data[metric.numerator_column.column_name],
                    data[metric.denominator_column.column_name],
                    data[covariate.numerator_column.column_name],
                    data[covariate.denominator_column.column_name],
                )
            )
        for covariate1 in processed_covariates:
            for covariate2 in processed_covariates:
                variance += (
                    covariate1.coef
                    * covariate2.coef
                    * get_delta_ratio_covariance(
                        data[covariate1.numerator_column.column_name],
                        data[covariate1.denominator_column.column_name],
                        data[covariate2.numerator_column.column_name],
                        data[covariate2.denominator_column.column_name],
                    )
                )
    return variance
