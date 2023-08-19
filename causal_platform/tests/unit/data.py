"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess

from causal_platform.tests.data import data_generator


def get_test_input(use_covariate=False):
    if use_covariate:
        raw_data = [
            [22323, "2019-01-01", 1, 0.34, "treatment", "ab_c", 1.1],
            [12345, "2019-01-02", 2, 0.13, "control", "ab_c", 1.2],
            [66663, "2019-01-02", 2, 0.19, "treatment", "ab_b", 1.2],
            [34346, "2019-01-02", 2, 0.13, "control", "ab_a", 1.3],
            [12123, "2019-01-02", 2, 0.05, "control", "ab_c", 0.9],
            [23234, "2019-01-02", 2, 0.11, "treatment", "ab_a", 1.4],
            [11112, "2019-01-02", 2, 0.3, "treatment", "ab_a", 1.3],
            [55555, "2019-01-02", 2, 0.23, "control", "ab_a", 1.1],
        ]
        data = pd.DataFrame(
            raw_data,
            columns=[
                "delivery_id",
                "date",
                "metric1",
                "metric2",
                "group",
                "cluster",
                "covariate1",
            ],
        )
        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "date": {"column_type": "date"},
                "delivery_id": {"column_type": "experiment_randomize_unit"},
                "metric1": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
                "metric2": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "log_transform": True,
                        "remove_outlier": True,
                        "check_distribution": False,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "quantile",
                        "quantile": 0.9,
                    },
                ],
                "covariate1": {
                    "column_type": "covariate",
                    "value_type": "numerical",
                },
            },
            "experiment_settings": {
                "is_check_flickers": True,
                "is_check_imbalance": True,
                "is_check_metric_type": True,
                "check_imbalance_method": "chi-square",
                "type": "ab",
            },
        }

    else:
        raw_data = [
            [22323, "2019-01-01", 1, 0.34, "treatment", "ab_c"],
            [12345, "2019-01-02", 2, 0.13, "control", "ab_c"],
            [66663, "2019-01-02", 2, 0.19, "treatment", "ab_b"],
            [34346, "2019-01-02", 2, 0.13, "control", "ab_a"],
            [12123, "2019-01-02", 2, 0.05, "control", "ab_c"],
            [23234, "2019-01-02", 2, 0.11, "treatment", "ab_a"],
            [11112, "2019-01-02", 2, 0.3, "treatment", "ab_a"],
            [55555, "2019-01-02", 2, 0.23, "control", "ab_a"],
        ]
        data = pd.DataFrame(
            raw_data,
            columns=["delivery_id", "date", "metric1", "metric2", "group", "cluster"],
        )
        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "date": {"column_type": "date"},
                "delivery_id": {"column_type": "experiment_randomize_unit"},
                "metric1": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
                "metric2": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "log_transform": True,
                        "remove_outlier": True,
                        "check_distribution": False,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "quantile",
                        "quantile": 0.9,
                    },
                ],
            },
            "experiment_settings": {
                "is_check_flickers": True,
                "is_check_imbalance": True,
                "is_check_metric_type": True,
                "type": "ab",
            },
        }
    data.date = pd.to_datetime(data.date)

    return data, config


def get_input_for_flicker_test():
    raw_data = [
        [22323, "2019-01-01", 1, 0.34, "treatment", "ab_c", 1.1],
        [22323, "2019-01-02", 2, 0.13, "control", "ab_c", 1.2],
        [66663, "2019-01-02", 2, 0.19, "treatment", "ab_b", 1.2],
        [66663, "2019-01-02", 2, 0.13, "control", "ab_a", 1.3],
        [12123, "2019-01-02", 2, 0.05, "control", "ab_c", 0.9],
        [12123, "2019-01-02", 2, 0.11, "treatment", "ab_a", 1.4],
        [11112, "2019-01-02", 2, 0.3, "treatment", "ab_a", 1.3],
        [11112, "2019-01-02", 2, 0.23, "control", "ab_a", 1.1],
        [22321, "2019-01-01", 1, 0.34, "treatment", "ab_c", 1.1],
        [22321, "2019-01-02", 2, 0.13, "control", "ab_c", 1.2],
        [66661, "2019-01-02", 2, 0.19, "treatment", "ab_b", 1.2],
        [66661, "2019-01-02", 2, 0.13, "control", "ab_a", 1.3],
        [12121, "2019-01-02", 2, 0.05, "control", "ab_c", 0.9],
        [12121, "2019-01-02", 2, 0.11, "treatment", "ab_a", 1.4],
        [11111, "2019-01-02", 2, 0.3, "treatment", "ab_a", 1.3],
        [11111, "2019-01-02", 2, 0.23, "control", "ab_a", 1.1],
        [22322, "2019-01-01", 1, 0.34, "treatment", "ab_c", 1.1],
        [22322, "2019-01-02", 2, 0.13, "control", "ab_c", 1.2],
        [66662, "2019-01-02", 2, 0.19, "treatment", "ab_b", 1.2],
        [66662, "2019-01-02", 2, 0.13, "control", "ab_a", 1.3],
        [12122, "2019-01-02", 2, 0.05, "control", "ab_c", 0.9],
        [23232, "2019-01-02", 2, 0.11, "treatment", "ab_a", 1.4],
        [11113, "2019-01-02", 2, 0.3, "treatment", "ab_a", 1.3],
        [55552, "2019-01-02", 2, 0.23, "control", "ab_a", 1.1],
    ]
    data = pd.DataFrame(
        raw_data,
        columns=[
            "delivery_id",
            "date",
            "metric1",
            "metric2",
            "group",
            "cluster",
            "covariate1",
        ],
    )
    config = {
        "columns": {
            "group": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
            },
            "date": {"column_type": "date"},
            "delivery_id": {"column_type": "experiment_randomize_unit"},
            "metric1": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
            },
        },
        "experiment_settings": {
            "is_check_flickers": True,
            "type": "ab",
        },
    }

    data.date = pd.to_datetime(data.date)

    return data, config


def get_ab_int_group_input(use_cov=False):
    if use_cov:
        raw_data = [
            [22323, "2019-01-01", 1, 0.34, 1.0, "ab_c", 1.1, 2.0],
            [12345, "2019-01-02", 2, 0.13, 0.0, "ab_c", 0.9, 1.0],
            [66663, "2019-01-02", 2, 0.19, 1.0, "ab_b", 0.4, 2.3],
            [34346, "2019-01-02", 2, 0.13, 0.0, "ab_a", 0.2, 2.2],
            [12123, "2019-01-02", 2, 0.05, 0.0, "ab_c", 0.4, 2.1],
            [23234, "2019-01-02", 2, 0.11, 1.0, "ab_a", 1.1, 1.4],
            [11112, "2019-01-02", 2, 0.3, 1.0, "ab_a", 0.1, 1.2],
            [55555, "2019-01-02", 2, 0.23, 0.0, "ab_a", 1.4, 1.3],
            [55555, "2019-01-02", 2, 0.13, 1.0, "ab_c", 1.2, 1.4],
        ]

        data = pd.DataFrame(
            raw_data,
            columns=[
                "delivery_id",
                "date",
                "metric1",
                "metric2",
                "group",
                "cluster",
                "cov1",
                "cov2",
            ],
        )

        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                    "variations_split": [0.5, 0.5],
                },
                "delivery_id": {"column_type": "experiment_randomize_unit"},
                "metric_ratio": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric1",
                    "denominator_column": "metric2",
                },
            },
            "experiment_settings": {"type": "ab"},
        }
    else:
        raw_data = [
            [22323, "2019-01-01", 1, 0.34, 1.0, "ab_c"],
            [12345, "2019-01-02", 2, 0.13, 0.0, "ab_c"],
            [66663, "2019-01-02", 2, 0.19, 1.0, "ab_b"],
            [34346, "2019-01-02", 2, 0.13, 0.0, "ab_a"],
            [12123, "2019-01-02", 2, 0.05, 0.0, "ab_c"],
            [23234, "2019-01-02", 2, 0.11, 1.0, "ab_a"],
            [11112, "2019-01-02", 2, 0.3, 1.0, "ab_a"],
            [55555, "2019-01-02", 2, 0.23, 0.0, "ab_a"],
            [55555, "2019-01-02", 2, 0.13, 1.0, "ab_c"],
        ]

        data = pd.DataFrame(
            raw_data,
            columns=["delivery_id", "date", "metric1", "metric2", "group", "cluster"],
        )

        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                    "variations_split": [0.5, 0.5],
                },
                "delivery_id": {"column_type": "experiment_randomize_unit"},
                "metric1": {"column_type": "metric", "metric_type": "continuous"},
            },
            "experiment_settings": {"type": "ab"},
        }
    return data, config


def get_missing_metric_input():
    raw_data = [
        ["2019-01-01", 1, np.nan, "treatment", "ab_c"],
        ["2019-01-02", 2, 0.13, "control", "ab_c"],
        ["2019-01-02", 2, np.nan, "treatment", "ab_b"],
        ["2019-01-02", 2, 0.13, "control", "ab_a"],
        ["2019-01-02", 3, np.nan, "control", "ab_c"],
        ["2019-01-02", 1, 0.11, "treatment", "ab_a"],
        ["2019-01-02", 2, 0.3, "treatment", "ab_a"],
        ["2019-01-02", 2, 0.23, "control", "ab_a"],
        ["2019-01-02", 2, 0.13, "control", "ab_c"],
    ]

    data = pd.DataFrame(raw_data, columns=["date", "metric1", "metric2", "group", "cluster"])
    data.date = pd.to_datetime(data.date)

    single_config = {
        "columns": {
            "group": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
            },
            "date": {"column_type": "date"},
            "metric2": [
                {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "check_distribution": False,
                }
            ],
            "metric1": [
                {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "check_distribution": False,
                }
            ],
        },
        "experiment_settings": {"type": "ab"},
    }
    return data, single_config


def get_config_with_customized_covariate():
    config_with_correct_applied_metrics = {
        "columns": {
            "group": {"column_type": "experiment_group"},
            "asap": [{"column_type": "metric", "metric_type": "continuous"}],
            "dat": [{"column_type": "metric", "metric_type": "continuous"}],
            "dr2": {"column_type": "metric", "metric_type": "continuous"},
            "pred_asap": {
                "column_type": "covariate",
                "applied_metrics": ["asap"],
                "value_type": "numerical",
            },
        },
        "experiment_settings": {"type": "ab"},
    }

    config_with_applied_metrics_covar_error = {
        "columns": {
            "group": {"column_type": "experiment_group"},
            "asap": [{"column_type": "metric", "metric_type": "continuous"}],
            "dat": [{"column_type": "metric", "metric_type": "continuous"}],
            "d2r": {"column_type": "metric", "metric_type": "continuous"},
            "pred_asap": {
                "column_type": "covariate",
                "applied_metrics": ["asap", "something_oops"],
                "value_type": "numerical",
            },
        },
        "experiment_settings": {"type": "ab"},
    }

    config_without_applied_metrics = {
        "columns": {
            "group": {"column_type": "experiment_group"},
            "asap": [{"column_type": "metric", "metric_type": "continuous"}],
            "dat": [{"column_type": "metric", "metric_type": "continuous"}],
            "d2r": {"column_type": "metric", "metric_type": "continuous"},
            "pred_asap": {"column_type": "covariate", "value_type": "numerical"},
        },
        "experiment_settings": {"type": "ab"},
    }
    return (
        config_with_correct_applied_metrics,
        config_with_applied_metrics_covar_error,
        config_without_applied_metrics,
    )


def get_quantile_test_input():
    raw_data = [
        ["2019-01-01", 1, 0.34, "treatment", "ab_c"],
        ["2019-01-02", 2, 0.13, "control", "ab_c"],
        ["2019-01-02", 3, 0.19, "treatment", "ab_b"],
        ["2019-01-02", 4, 0.13, "control", "ab_a"],
        ["2019-01-02", 5, 0.05, "control", "ab_c"],
        ["2019-01-02", 6, 0.11, "treatment", "ab_a"],
        ["2019-01-02", 7, 0.3, "treatment", "ab_a"],
        ["2019-01-02", 8, 0.23, "control", "ab_a"],
        ["2019-01-02", 9, 0.13, "control", "ab_c"],
        ["2019-01-02", 10, 0.13, "control", "ab_c"],
    ]

    data = pd.DataFrame(raw_data, columns=["date", "metric1", "metric2", "group", "cluster"])
    data.date = pd.to_datetime(data.date)

    config = {
        "columns": {
            "group": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
            },
            "date": {"column_type": "date"},
            "metric1": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "quantile",
                "quantile": 0.9,
            },
        },
        "experiment_settings": {
            "is_check_flickers": False,
            "is_check_metric_type": False,
            "type": "ab",
            "bootstrap_size": 9,
            "bootstrap_iteration": 1000,
        },
    }
    return data, config


def get_redundant_columns_table_input():
    raw_data = [
        ["2019-01-01", 1, 0.34, "treatment", "ab_c", "ants", 31],
        ["2019-01-02", 2, 0.13, "control", "ab_c", "bees", 32],
        ["2019-01-02", 3, 0.19, "treatment", "ab_b", "beetles", 33],
        ["2019-01-02", 4, 0.13, "control", "ab_a", "butterflies", 34],
        ["2019-01-02", 5, 0.05, "control", "ab_c", "caddisflies", 35],
        ["2019-01-02", 6, 0.11, "treatment", "ab_a", "cockroaches", 36],
        ["2019-01-02", 7, 0.3, "treatment", "ab_a", "crickets", 37],
        ["2019-01-02", 8, 0.23, "control", "ab_a", "diplurans", 38],
        ["2019-01-02", 9, 0.13, "control", "ab_c", "dragonflies", 39],
        ["2019-01-02", 10, 0.13, "control", "ab_c", "damselflies", 40],
    ]

    data = pd.DataFrame(
        raw_data,
        columns=["date", "metric1", "metric2", "group", "cluster", "insects", "num_of_years"],
    )
    data.date = pd.to_datetime(data.date)

    config = {
        "columns": {
            "group": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
            },
            "date": {"column_type": "date"},
            "metric1": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "quantile",
                "quantile": 0.9,
            },
        },
        "experiment_settings": {
            "is_check_flickers": False,
            "is_check_metric_type": False,
            "type": "ab",
            "bootstrap_size": 9,
            "bootstrap_iteration": 1000,
        },
    }
    return data, config


def get_preprocess_only_test_input():
    raw_data = [
        ["a", "treatment"],
        ["a", "treatment"],
        ["a", "control"],
        ["b", "treatment"],
        ["b", "treatment"],
        ["c", "control"],
        ["c", "control"],
        ["d", "treatment"],
        ["d", "treatment"],
        ["e", "control"],
        ["e", "control"],
    ]
    data = pd.DataFrame(
        raw_data,
        columns=["user", "group"],
    )
    config_bypass_check = {
        "columns": {
            "group": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
            },
            "user": {"column_type": "experiment_randomize_unit"},
        },
        "experiment_settings": {
            "is_check_imbalance": False,
            "is_check_flickers": False,
            "type": "ab",
        },
    }

    config = {
        "columns": {
            "group": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
            },
            "user": {"column_type": "experiment_randomize_unit"},
        },
        "experiment_settings": {
            "type": "ab",
        },
    }

    return data, config, config_bypass_check


def get_ratio_test_input(iterations=1000, use_cov=False):
    if use_cov:
        raw_data = [
            ["2019-01-01", 1, 4, "treatment", "ab_c", 1.1, 0.9],
            ["2019-01-02", 1, 3, "control", "ab_c", 0.2, 0.3],
            ["2019-01-02", 1, 1, "treatment", "ab_b", 0.5, 0.6],
            ["2019-01-02", 1, 0, "control", "ab_a", 0.1, 0.9],
            ["2019-01-02", 1, 0, "control", "ab_c", 1.2, 1.2],
            ["2019-01-02", 1, 8, "treatment", "ab_a", 1.0, 1.1],
            ["2019-01-02", 1, 3, "treatment", "ab_a", 0.2, 0.3],
            ["2019-01-02", 1, 6, "control", "ab_a", 0.0, 2.3],
            ["2019-01-02", 2, 3, "control", "ab_c", 1.2, 1.5],
            ["2019-01-02", 1, 3, "control", "ab_c", 0.3, 0.5],
        ]
        data = pd.DataFrame(
            raw_data,
            columns=["date", "metric1", "metric2", "group", "cluster", "covariate1", "covariate2"],
        )

        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "date": {"column_type": "date"},
                "a_ratio_metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric2",
                    "denominator_column": "metric1",
                },
                "a_ratio_covariate": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "covariate1",
                    "denominator_column": "covariate2",
                },
            },
            "experiment_settings": {
                "is_check_imbalance": False,
                "is_check_flickers": False,
                "is_check_metric_type": False,
                "type": "ab",
                "bootstrap_size": 9,
                "bootstrap_iteration": iterations,
            },
        }
    else:
        raw_data = [
            ["2019-01-01", 1, 4, "treatment", "ab_c"],
            ["2019-01-02", 1, 3, "control", "ab_c"],
            ["2019-01-02", 1, 1, "treatment", "ab_b"],
            ["2019-01-02", 1, 0, "control", "ab_a"],
            ["2019-01-02", 1, 0, "control", "ab_c"],
            ["2019-01-02", 1, 8, "treatment", "ab_a"],
            ["2019-01-02", 1, 3, "treatment", "ab_a"],
            ["2019-01-02", 1, 6, "control", "ab_a"],
            ["2019-01-02", 2, 3, "control", "ab_c"],
            ["2019-01-02", 1, 3, "control", "ab_c"],
        ]
        data = pd.DataFrame(raw_data, columns=["date", "metric1", "metric2", "group", "cluster"])
        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "date": {"column_type": "date"},
                "a_ratio_metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric2",
                    "denominator_column": "metric1",
                },
            },
            "experiment_settings": {
                "is_check_imbalance": False,
                "is_check_flickers": False,
                "is_check_metric_type": False,
                "type": "ab",
                "bootstrap_size": 9,
                "bootstrap_iteration": iterations,
            },
        }
    data.date = pd.to_datetime(data.date)

    return data, config


def get_diff_in_diff_input():
    data = [
        ["2019-01-01", 1, 100, 0.5],
        ["2019-01-02", 1, 70, 0.4],
        ["2019-01-03", 1, 120, 0.7],
        ["2019-01-04", 1, 140, 0.8],
        ["2019-01-05", 1, 90, 0.1],
        ["2019-01-06", 1, 10, 0.3],
        ["2019-01-07", 1, 140, 0.2],
        ["2019-01-08", 1, 140, 0.5],
        ["2019-01-09", 1, 100, 0.1],
        ["2019-01-10", 1, 10, 0.2],
        ["2019-01-11", 1, 30, 0.3],
        ["2019-01-12", 1, 60, 0.4],
        ["2019-01-01", 2, 10, 0.5],
        ["2019-01-02", 2, 20, 0.4],
        ["2019-01-03", 2, 10, 0.7],
        ["2019-01-04", 2, 40, 0.8],
        ["2019-01-05", 2, 190, 0.1],
        ["2019-01-06", 2, 120, 0.3],
        ["2019-01-07", 2, 10, 0.2],
        ["2019-01-08", 2, 40, 0.5],
        ["2019-01-09", 2, 90, 0.1],
        ["2019-01-10", 2, 20, 0.2],
        ["2019-01-11", 2, 60, 0.3],
        ["2019-01-12", 2, 90, 0.4],
        ["2019-01-01", 3, 10, 0.5],
        ["2019-01-02", 3, 20, 0.4],
        ["2019-01-03", 3, 10, 0.7],
        ["2019-01-04", 3, 40, 0.8],
        ["2019-01-05", 3, 190, 0.1],
        ["2019-01-06", 3, 120, 0.3],
        ["2019-01-07", 3, 10, 0.2],
        ["2019-01-08", 3, 40, 0.5],
        ["2019-01-09", 3, 90, 0.1],
        ["2019-01-10", 3, 20, 0.2],
        ["2019-01-11", 3, 60, 0.3],
        ["2019-01-12", 3, 90, 0.4],
    ]
    df = pd.DataFrame(data, columns=["date", "market", "applicant", "cvr"])
    df.date = pd.to_datetime(df.date)

    config = {
        "columns": {
            "market": {"column_type": "experiment_randomize_unit"},
            "date": {"column_type": "date"},
            "applicant": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": True,
                "remove_outlier": True,
                "check_distribution": False,
            },
        },
        "experiment_settings": {
            "treatment_unit_ids": [1],
            "match_unit_size": 5,
            "exclude_unit_ids": [3],
            "matching_method": "correlation",
            "matching_start_date": "2019-01-01",
            "matching_end_date": "2019-01-04",
            "experiment_start_date": "2019-01-05",
            "experiment_end_date": "2019-01-12",
            "matching_columns": ["applicant", "cvr"],
            "matching_weights": [0.5, 0.5],
            "type": "diff_in_diff",
        },
    }
    return df, config


def get_real_diff_in_diff_input():
    df = data_generator.get_sample_data()
    df["applied_date"] = pd.to_datetime(df["applied_date"])

    config = {
        "columns": {
            "applied_submarket_id": {"column_type": "experiment_randomize_unit"},
            "applied_date": {"column_type": "date"},
            "count_of_applicants": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": True,
                "remove_outlier": True,
                "check_distribution": False,
            },
        },
        "experiment_settings": {
            "treatment_unit_ids": [1, 4, 6],
            "match_unit_size": 5,
            "exclude_unit_ids": [3],
            "matching_method": "correlation",
            "matching_start_date": "2019-09-10",
            "matching_end_date": "2019-09-15",
            "experiment_start_date": "2019-09-16",
            "experiment_end_date": "2019-09-19",
            "matching_columns": ["count_of_applicants"],
            "matching_weights": [1],
            "type": "diff_in_diff",
        },
    }
    return df, config


def get_small_sample_diff_in_diff_input():
    df = data_generator.get_sample_data()
    df["applied_date"] = pd.to_datetime(df["applied_date"])

    config_no_adjust = {
        "columns": {
            "applied_submarket_id": {"column_type": "experiment_randomize_unit"},
            "applied_date": {"column_type": "date"},
            "count_of_applicants": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": True,
                "remove_outlier": True,
                "check_distribution": False,
            },
        },
        "experiment_settings": {
            "treatment_unit_ids": [1, 4],
            "match_unit_size": 2,
            "exclude_unit_ids": [3],
            "matching_method": "correlation",
            "matching_start_date": "2019-09-10",
            "matching_end_date": "2019-09-15",
            "experiment_start_date": "2019-09-16",
            "experiment_end_date": "2019-09-19",
            "matching_columns": ["count_of_applicants"],
            "matching_weights": [0.5, 0.5],
            "small_sample_adjustment": False,
            "type": "diff_in_diff",
        },
    }

    config_with_adjust = {
        "columns": {
            "applied_submarket_id": {"column_type": "experiment_randomize_unit"},
            "applied_date": {"column_type": "date"},
            "count_of_applicants": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": True,
                "remove_outlier": True,
                "check_distribution": False,
            },
        },
        "experiment_settings": {
            "treatment_unit_ids": [1, 4],
            "match_unit_size": 2,
            "exclude_unit_ids": [3],
            "matching_method": "correlation",
            "matching_start_date": "2019-09-10",
            "matching_end_date": "2019-09-15",
            "experiment_start_date": "2019-09-16",
            "experiment_end_date": "2019-09-19",
            "matching_columns": ["count_of_applicants"],
            "matching_weights": [0.5, 0.5],
            "type": "diff_in_diff",
        },
    }

    return df, config_no_adjust, config_with_adjust


def get_imbalance_check_input():
    raw_data = [
        ["2019-01-01", 1, 0.34, "treatment", "A", "ab_c"],
        ["2019-01-02", 2, 0.13, "control", "A", "ab_c"],
        ["2019-01-02", 2, 0.19, "treatment", "A", "ab_b"],
        ["2019-01-02", 2, 0.13, "control", "A", "ab_a"],
        ["2019-01-02", 2, 0.05, "control", "A", "ab_c"],
        ["2019-01-02", 2, 0.11, "treatment", "B", "ab_a"],
        ["2019-01-02", 2, 0.3, "treatment", "A", "ab_a"],
        ["2019-01-02", 2, 0.23, "control", "A", "ab_a"],
        ["2019-01-02", 2, 0.13, "control", "A", "ab_d"],
        ["2019-01-02", 2, 0.13, "control", "A", "ab_e"],
        ["2019-01-02", 2, 0.13, "control", "A", "ab_f"],
        ["2019-01-02", 2, 0.13, "control", "A", "ab_g"],
        ["2019-01-02", 2, 0.13, "control", "A", "ab_h"],
    ]

    data = pd.DataFrame(raw_data, columns=["date", "metric1", "metric2", "group1", "group2", "cluster"])
    data.date = pd.to_datetime(data.date)

    config = {
        "columns": {
            "group1": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.1, 0.9],
            },
            "group2": {
                "column_type": "experiment_group",
                "control_label": "A",
                "variations": ["A", "B"],
                "variations_split": [0.5, 0.5],
            },
            "date": {"column_type": "date"},
            "metric1": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": True,
                "remove_outlier": True,
                "check_distribution": False,
            },
            "metric2": [
                {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
                {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "quantile",
                    "quantile": 0.9,
                },
            ],
        },
        "experiment_settings": {
            "is_check_imbalance": True,
            "is_check_flickers": False,
            "is_check_metric_type": True,
            "check_imbalance_method": "binomial",
            "type": "ab",
        },
    }

    config_w_cluster = {
        "columns": {
            "group1": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.1, 0.9],
            },
            "group2": {
                "column_type": "experiment_group",
                "control_label": "A",
                "variations": ["A", "B"],
                "variations_split": [0.5, 0.5],
            },
            "metric1": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": True,
                "remove_outlier": True,
                "check_distribution": False,
            },
            "cluster": {"column_type": "cluster"},
        },
        "experiment_settings": {
            "is_check_imbalance": True,
            "is_check_flickers": False,
            "check_imbalance_method": "chi-square",
            "is_check_metric_type": True,
            "type": "ab",
        },
    }
    return data, config, config_w_cluster


def get_preprocess_test_input():
    raw_data = [
        [22323, 10, "2019-01-01", 1, 0.34, "treatment", "A", "ab_c"],
        [12123, 14, "2019-01-02", 2, 0.05, "control", "B", "ab_c"],
        [11112, 16, "2019-01-02", 2, 0.3, "treatment", "A", "ab_a"],
        [11112, 16, "2019-01-03", 6, 0.13, "treatment", "A", "ab_c"],
        [55555, 17, "2019-01-03", 6, 0.13, "control", "A", "ab_c"],
        [55555, 13, "2019-01-02", 2, 0.13, "control", "A", "ab_a"],
        [22323, 10, "2019-01-01", 1, 0.34, "treatment", "A", "ab_c"],
        [12123, 14, "2019-01-02", 2, 0.05, "control", "B", "ab_c"],
        [55555, 17, "2019-01-03", 6, 0.13, "control", "A", "ab_c"],
        [55555, 13, "2019-01-02", 2, 0.13, "control", "A", "ab_a"],
        # Flickers:
        [12345, 11, "2019-01-02", 2, 0.13, "control", "B", "ab_c"],
        [12345, 11, "2019-01-03", 2, 0.19, "treatment", "B", "ab_b"],
        [23234, 15, "2019-01-02", 2, 0.11, "treatment", "B", "ab_a"],
        [23234, 15, "2019-01-02", 2, 0.11, "control", "B", "ab_a"],
        [98979, 17, "2019-01-02", 2, 0.23, "control", "B", "ab_a"],
        [98979, 17, "2019-01-02", 2, 0.13, "control", "A", "ab_c"],
        [12345, 11, "2019-01-02", 2, 0.13, "control", "B", "ab_c"],
        [12345, 11, "2019-01-03", 2, 0.19, "treatment", "B", "ab_b"],
        [23234, 15, "2019-01-02", 2, 0.11, "treatment", "B", "ab_a"],
        [23234, 15, "2019-01-02", 2, 0.11, "control", "B", "ab_a"],
        [98979, 17, "2019-01-02", 2, 0.23, "control", "B", "ab_a"],
        [98979, 17, "2019-01-02", 2, 0.13, "control", "A", "ab_c"],
    ]

    data = pd.DataFrame(
        raw_data,
        columns=[
            "market_id",
            "hour",
            "date",
            "metric1",
            "metric2",
            "group1",
            "group2",
            "cluster",
        ],
    )
    data.date = pd.to_datetime(data.date)

    config = {
        "columns": {
            "group1": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
            },
            "group2": {
                "column_type": "experiment_group",
                "control_label": "A",
                "variations": ["A", "B"],
                "variations_split": [0.5, 0.5],
            },
            "date": {"column_type": "date"},
            "market_id": {"column_type": "experiment_randomize_unit"},
            "hour": {"column_type": "experiment_randomize_unit"},
            "metric1": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": False,
                "remove_outlier": True,
                "check_distribution": True,
            },
            "metric2": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
                "log_transform": False,
                "remove_outlier": True,
                "check_distribution": True,
            },
            "cluster": {"column_type": "cluster"},
        },
        "experiment_settings": {
            "is_check_imbalance": True,
            "is_check_flickers": True,
            "is_remove_flickers": True,
            "is_check_metric_type": True,
            "type": "ab",
        },
    }
    # TODO: when treatment/control is not 50/50
    return data, config


def get_ab_input_with_flicker_and_imbalance():
    raw_data = [
        [22323, "2019-01-01", 1, 0.34, "treatment", "ab_c"],
        [66663, "2019-01-02", 2, 0.19, "treatment", "ab_b"],
        [34346, "2019-01-02", 2, 0.13, "treatment", "ab_a"],
        [12123, "2019-01-02", 2, 0.05, "treatment", "ab_c"],
        [23234, "2019-01-02", 2, 0.11, "treatment", "ab_e"],
        [23235, "2019-01-02", 2, 0.11, "treatment", "ab_f"],
        [23236, "2019-01-02", 2, 0.11, "treatment", "ab_g"],
        [23237, "2019-01-02", 2, 0.11, "treatment", "ab_h"],
        [23238, "2019-01-02", 2, 0.11, "treatment", "ab_g"],
        [12345, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [12346, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [12347, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [12348, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [12349, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [12350, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [12351, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [12352, "2019-01-02", 2, 0.13, "treatment", "ab_i"],
        [11112, "2019-01-02", 2, 0.3, "control", "ab_a"],
        [11113, "2019-01-02", 2, 0.32, "control", "ab_a"],
        [55555, "2019-01-02", 2, 0.23, "control", "ab_a"],
        [55555, "2019-01-02", 2, 0.23, "treatment", "ab_a"],
    ]

    data = pd.DataFrame(
        raw_data,
        columns=["delivery_id", "date", "metric1", "metric2", "group", "cluster"],
    )
    data.date = pd.to_datetime(data.date)

    config = {
        "columns": {
            "group": {
                "column_type": "experiment_group",
                "control_label": "control",
                "variations": ["control", "treatment"],
                "variations_split": [0.5, 0.5],
                "check_imbalance_method": "chi-square",
            },
            "date": {"column_type": "date"},
            "delivery_id": {"column_type": "experiment_randomize_unit"},
            "metric1": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
            },
            "metric2": {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
            },
        },
        "experiment_settings": {
            "is_check_flickers": True,
            "is_check_imbalance": True,
            "is_check_metric_type": True,
            "is_remove_flickers": True,
            "type": "ab",
        },
    }
    return data, config


def generate_ar_seq_for_one_group(ar_cor, group_size):
    ar = np.array([1, -1 * ar_cor])
    ma = np.array([1])
    AR_object = ArmaProcess(ar, ma)
    w = AR_object.generate_sample(nsample=group_size)
    return pd.Series(w)
