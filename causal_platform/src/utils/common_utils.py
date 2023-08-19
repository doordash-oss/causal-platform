"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import math
from json import JSONEncoder
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from causal_platform.src.models.configuration_model.base_objects import (
    Covariate,
    CovariateType,
    DateColumn,
    Metric,
    MetricType,
)
from causal_platform.src.models.data_model import DataLoader
from causal_platform.src.models.message.message import Message, MessageCollection, Source, Status
from causal_platform.src.utils.logger import logger
from causal_platform.src.utils.validation_utils import check_data_is_object_type


def get_all_attr_of_class(cls) -> List[str]:
    return [t for i, t in cls.__dict__.items() if i[:1] != "_"]


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Message) or isinstance(obj, MessageCollection):
            return obj.__dict__
        elif isinstance(obj, Status) or isinstance(obj, Source):
            return obj.name if obj is not None else ""
        else:
            return super(NumpyEncoder, self).default(obj)


def convert_table_column_to_lower_case(df: Union[pd.DataFrame, DataLoader]):
    df.columns = [str.lower(str(col)) for col in df.columns]


def convert_data_to_proper_types(
    df: Union[pd.DataFrame, DataLoader],
    metrics: List[Metric],
    date: Optional[DateColumn] = None,
    covariates: Optional[List[Covariate]] = None,
):

    column_names = []
    for metric in metrics:
        if metric.metric_type == MetricType.ratio:
            column_names += [metric.numerator_column.column_name]
            column_names += [metric.denominator_column.column_name]
        else:
            column_names += [metric.column_name]

    if covariates is not None:
        for covariate in covariates:
            if covariate.value_type == CovariateType.ratio:
                column_names += [
                    covariate.numerator_column.column_name,
                    covariate.denominator_column.column_name,
                ]
            elif covariate.value_type == CovariateType.numerical:
                column_names += [covariate.column_name]

    if isinstance(df, pd.DataFrame):
        for column_name in column_names:
            try:
                df[column_name] = df[column_name].astype(float, errors="ignore")
            except Exception:
                logger.info(f"Convert {column_name} metric to float dtype failed.")

        if date is not None:
            column_name = date.column_name
            if check_data_is_object_type(df, column_name):
                if date.date_format is not None:
                    df[column_name] = pd.to_datetime(df[column_name], format=date.date_format)
                else:
                    df[column_name] = pd.to_datetime(df[column_name])

    elif isinstance(df, DataLoader):
        if date is None:
            date_columns = []
            date_formats = []
        else:
            date_columns = [date.column_name]
            date_formats = [] if date.date_format is None else [date.format]

        df.force_cols_to_number_or_date(num_columns=column_names, date_columns=date_columns, date_formats=date_formats)


def isnan(value):
    # math.isnan can't take NoneType data
    if value is None:
        return True
    if math.isnan(value):
        return True
    return False


def format_number(num, decimals, format="float") -> str:
    assert format in ["float", "percent", "exp"]
    format_template = ":." + str(decimals)
    if format == "float":
        format_template += "f"
    if format == "percent":
        format_template += "%"
    if format == "exp":
        format_template += "e"
    format_template = "{" + format_template + "}"
    return format_template.format(num)
