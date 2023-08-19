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

from causal_platform.src.models.configuration_model.base_objects import ColumnType, MetricType, TColumn
from causal_platform.src.models.data_model import DataLoader


def check_column_in_data(data: Union[pd.DataFrame, DataLoader], column_name: str) -> bool:
    return column_name in data.columns


def check_column_is_type(data: Union[pd.DataFrame, DataLoader], column: TColumn, type: str) -> bool:
    if type == MetricType.continuous:
        return check_data_is_continuous(data, column.column_name)
    elif type == MetricType.proportional:
        return check_data_is_proportional(data, column.column_name)
    elif type == MetricType.ratio:
        return True
    elif type == ColumnType.date and column.column_type == ColumnType.date:
        return check_data_is_datetime(data, column.column_name)
    return False


def check_data_is_continuous(data: Union[pd.DataFrame, DataLoader], column_name: str) -> bool:

    # TODO: Clear this out when proportional logic is removed. Skip proportional check bc it will be deprecated.
    if isinstance(data, DataLoader):
        return np.issubdtype(data.dtypes[column_name], np.number)

    # when dtype of the column is object, try convert it to float. If it's not
    # numeric data, it will return the original dtype
    return np.issubdtype(data.dtypes[column_name], np.number)


def check_data_is_proportional(data: Union[pd.DataFrame, DataLoader], column_name: str) -> bool:

    # TODO: Clear this out when proportional logic is removed. Skip proportional check bc it will be deprecated.
    if isinstance(data, DataLoader):
        return np.issubdtype(data.dtypes[column_name], np.number)

    column_data = data[column_name].dropna()
    return (
        np.issubdtype(column_data.dtype, np.number)
        and bool(np.isin([0, 1], column_data.unique()).all())
        and len(column_data.unique()) == 2
    )


def check_data_is_datetime(data: Union[pd.DataFrame, DataLoader], column_name: str) -> bool:
    return np.issubdtype(data.dtypes[column_name], np.datetime64)


def check_data_is_object_type(data: Union[pd.DataFrame, DataLoader], column_name: str) -> bool:
    return np.issubdtype(data.dtypes[column_name], np.object_)
