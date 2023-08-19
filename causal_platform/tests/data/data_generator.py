"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def get_ab_test_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__)) + "/ab_test_data.csv")


def get_causal_test_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__)) + "/causal_test_data.csv")


def get_sample_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__)) + "/sample_data.csv")


def get_multi_ratio_cov_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__)) + "/multi_ratio_cov_data.csv")


def generate_ar_seq_for_one_group(ar_cor, group_size):
    ar = np.array([1, -1 * ar_cor])
    ma = np.array([1])
    AR_object = ArmaProcess(ar, ma)
    w = AR_object.generate_sample(nsample=group_size)
    return pd.Series(w)


def generate_ab_data(
    group_mean=10,
    group_std=1,
    ar_cor=0.8,
    total_group=700,
    time_length=30,
    treatment_effect=5,
    error_mean=0,
    error_std=1,
):
    # add group error and AR error
    df_group = pd.DataFrame(
        {
            "group": range(total_group),
            "group_error": np.random.normal(group_mean, group_std, total_group),
        }
    )
    temp = pd.concat(
        [
            df_group,
            df_group.apply(lambda x: generate_ar_seq_for_one_group(ar_cor, time_length), axis=1),
        ],
        axis=1,
    )
    df = pd.melt(
        temp,
        id_vars=["group", "group_error"],
        value_vars=list(temp.columns[2:]),
        var_name="time_index",
        value_name="ar_error",
    )
    total_treatment_group = int(total_group / 2)
    df["is_treatment_group"] = df["group"].isin(range(total_treatment_group)) * 1

    df["asap"] = (
        treatment_effect * df["is_treatment_group"]
        + df["group_error"]
        + np.random.normal(error_mean, error_std, df.shape[0])
    )

    return df
