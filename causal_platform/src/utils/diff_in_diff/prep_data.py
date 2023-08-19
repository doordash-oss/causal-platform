"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import datetime
from typing import Callable, List

import pandas as pd

from causal_platform.src.utils.constants import Constants


def get_data_filtered_by_unit_ids(data: pd.DataFrame, unit_column_name: str, unit_ids: List[int]) -> pd.DataFrame:
    return data[data[unit_column_name].isin(unit_ids)]


def get_data_between_start_end_date(
    data: pd.DataFrame,
    date_column_name: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> pd.DataFrame:
    return data[(data[date_column_name] >= start_date) & (data[date_column_name] <= end_date)]


def aggregate_metric_by_date(
    data: pd.DataFrame,
    date_column_name: str,
    metric_column_name: str,
    aggregate_func: Callable,
) -> pd.Series:
    return data.groupby(date_column_name)[metric_column_name].apply(aggregate_func)


def get_aggregate_metric_in_unit_ids(
    data: pd.DataFrame,
    date_column_name: str,
    unit_column_name: str,
    unit_ids: List[int],
    metric_column_name: str,
    aggregate_func: Callable,
) -> pd.Series:

    data_in_units = get_data_filtered_by_unit_ids(data, unit_column_name, unit_ids)
    agg_metric_by_date = aggregate_metric_by_date(data_in_units, date_column_name, metric_column_name, aggregate_func)

    return agg_metric_by_date


def get_unit_candidates(
    data: pd.DataFrame,
    unit_column_name: str,
    exclude_unit_ids: List[int],
    treatment_unit_ids: List[int],
) -> List[int]:
    """function to return regions that are eligible as control region

    Returns:
        List[int] -- list of unique region ids
    """
    all_unit_ids = data[unit_column_name].unique()
    unit_candidates = list(set(all_unit_ids) - set(exclude_unit_ids) - set(treatment_unit_ids))
    return unit_candidates


def prep_data_for_diff_in_diff(
    data: pd.DataFrame,
    treatment_unit_ids: List[int],
    control_unit_ids: List[int],
    unit_column_name: str,
    date_column_name: str,
    treatment_start_date,
) -> pd.DataFrame:
    experiment_data = data[data[unit_column_name].isin(treatment_unit_ids + control_unit_ids)].copy()
    experiment_data.loc[:, Constants.DIFF_IN_DIFF_TREATMENT] = experiment_data[unit_column_name].apply(
        lambda x: Constants.DIFF_IN_DIFF_TREATMENT_VALUE
        if x in treatment_unit_ids
        else Constants.DIFF_IN_DIFF_CONTROL_VALUE
    )
    experiment_data.loc[:, Constants.DIFF_IN_DIFF_TIME] = experiment_data[date_column_name].apply(
        lambda x: Constants.DIFF_IN_DIFF_TIME_AFTER if x >= treatment_start_date else Constants.DIFF_IN_DIFF_TIME_BEFORE
    )
    return experiment_data
