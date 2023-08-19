"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Union

import numpy as np
import pandas as pd

from causal_platform.src.models.configuration_model.base_objects import MatchingMethod
from causal_platform.src.models.configuration_model.config import DiDConfig
from causal_platform.src.utils.error import InputDataError


def calculate_correlation(series1: Union[np.array, pd.Series], series2: Union[np.array, pd.Series]):
    """function to calculate correlation of two series

    Arguments:
        series1 {Union[np.array, pd.Series]} -- (n, 1) array
        series2 {Union[np.array, pd.Series]} -- (n, 1) array

    Returns:
        [float] -- correlation coefficient
    """
    if (len(series1.shape) > 1 and series1.shape[1] > 1) or (len(series1.shape) > 1 and series2.shape[1] > 1):
        raise InputDataError("Input array is not valid")

    return np.corrcoef(series1, series2)[0, 1]


def calculate_euclidean_distance(series1: np.array, series2: np.array):
    """function to calculate euclidean distance of two array

    Arguments:
        series1 {np.array} -- (n, 1) array
        series2 {np.array} -- (n, 1) array

    Returns:
        distance [int] -- euclidean distance of series1 and series2
    """
    diff = series1 - series2
    distance = np.sqrt(np.dot(diff.T, diff))
    return distance


def calculate_distance(series1, series2, config):
    if config.matching_method == MatchingMethod.correlation:
        distance = -calculate_correlation(series1, series2)

    # TODO(caixia): implement other methods
    return distance


def calculate_standardize_metric(data: np.array) -> np.array:
    """standardize metric

    Arguments:
        data {np.array} -- array that contains the columns to be standardized
    """
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))


def calculate_weighted_sum(data: np.array, weights: np.array) -> np.array:
    """calculate weighted sum of the matching metrics

    Arguments:
        data {np.array} -- (n, k) array (k = number of matching metrics)
        weights {np.array} -- (k, ) or (1, k) array

    Returns:
        np.array -- (n, ) array
    """
    return (data * weights).sum(axis=1)


def standardize_and_calculate_weighted_sum(data: np.array, config: DiDConfig) -> np.array:
    if len(config.matching_columns) == 1:
        # if there is only one matching column, don't process it
        weighted_sum = data
    else:
        standardized_array = calculate_standardize_metric(data)
        weighted_sum = calculate_weighted_sum(standardized_array, np.array(config.matching_weights))
    return weighted_sum
