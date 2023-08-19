"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from causal_platform.src.models.configuration_model.base_objects import (
    Column,
    ColumnType,
    Covariate,
    CovariateType,
    ExperimentGroup,
    ExperimentVariation,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.utils.constants import Constants
from causal_platform.tests.data import data_generator


class ABTestBase:
    @pytest.fixture
    def metrics(self) -> List[Metric]:
        return [
            Metric(
                "ASAP",
                MetricType.continuous,
                MetricAggregateFunc.mean,
                False,
                False,
                False,
            ),
            Metric(
                "DAT",
                MetricType.continuous,
                MetricAggregateFunc.mean,
                False,
                False,
                False,
            ),
        ]

    @pytest.fixture
    def metrics_map(self) -> Dict[str, Metric]:
        return {
            "ASAP": Metric(
                "ASAP",
                MetricType.continuous,
                MetricAggregateFunc.mean,
                False,
                False,
                False,
            ),
            "DAT": Metric(
                "DAT",
                MetricType.continuous,
                MetricAggregateFunc.mean,
                False,
                False,
                False,
            ),
        }

    @pytest.fixture
    def covariates(self) -> List[Covariate]:
        return [
            Covariate("FLF", CovariateType.numerical),
            Covariate("SP_ID", CovariateType.categorial),
        ]

    @pytest.fixture
    def covariates_map(self) -> Dict[str, Covariate]:
        return {
            "flf": Covariate("FLF", CovariateType.numerical),
            "sp_id": Covariate("SP_ID", CovariateType.categorial),
        }

    @pytest.fixture
    def cluster_column(self) -> Column:
        return Column("UNIT_ID", ColumnType.cluster)

    @pytest.fixture
    def experiment_group1(self):
        return ExperimentGroup(
            "GROUP1",
            control=ExperimentVariation("control", 0.5),
            treatments=[ExperimentVariation("treatment", 0.5)],
        )

    @pytest.fixture
    def experiment_group2(self):
        return ExperimentGroup(
            "GROUP2",
            control=ExperimentVariation(0, 0.5),
            treatments=[ExperimentVariation(1, 0.5)],
        )

    @pytest.fixture
    def experiment_groups(self, experiment_group1, experiment_group2) -> List[ExperimentGroup]:
        return [experiment_group1, experiment_group2]

    @pytest.fixture
    def experiment_groups_map(self, experiment_group1, experiment_group2) -> Dict[str, ExperimentGroup]:
        return {"GROUP1": experiment_group1, "GROUP2": experiment_group2}

    @pytest.fixture
    def interactions(self, experiment_group1, experiment_group2) -> List[Tuple[Column, Column]]:
        return [
            (experiment_group1, experiment_group2),
            (experiment_group1, Covariate("SP_ID", CovariateType.categorial)),
        ]

    @pytest.fixture
    def ab_config_dict(self, experiment_group1, experiment_group2) -> Dict:
        return {
            "columns": {
                "GROUP1": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "GROUP2": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                    "variations_split": [0.5, 0.5],
                },
                "SP_ID": {"column_type": "covariate", "value_type": "categorical"},
                "FLF": {"column_type": "covariate", "value_type": "numerical"},
                "UNIT_ID": {"column_type": "cluster"},
                "ASAP": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": False,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
                "DAT": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": False,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
            },
            "experiment_settings": {
                "is_check_imbalance": False,
                "is_check_flickers": False,
                "is_check_metric_type": True,
                "type": "ab",
            },
        }

    @pytest.fixture
    def quantile_config(data) -> Dict:
        return {
            "columns": {
                "GROUP1": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "SP_ID": {"column_type": "covariate", "value_type": "categorical"},
                "FLF": {"column_type": "covariate", "value_type": "numerical"},
                "UNIT_ID": {"column_type": "cluster"},
                "ASAP": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "quantile",
                    "quantile": 0.95,
                    "log_transform": False,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
                "DAT": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": False,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
            },
            "experiment_settings": {
                "is_check_imbalance": False,
                "is_check_flickers": False,
                "is_check_metric_type": True,
                "type": "ab",
            },
        }

    @pytest.fixture
    def config_without_customized_covar(self):
        config_without_customized_covar = {
            "columns": {
                "GROUP1": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "ASAP": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "fitter_type": "regression",
                },
                "DAT": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "fitter_type": "regression",
                },
                "SUBMARKET_ID": {"column_type": "covariate", "value_type": "categorical"},
                "UNIT_ID": {"column_type": "cluster"},
            },
            "experiment_settings": {"type": "ab"},
        }
        return config_without_customized_covar

    @pytest.fixture
    def config_with_customized_covar(self):
        config_with_customized_covar = {
            "columns": {
                "GROUP1": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "ASAP": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "fitter_type": "regression",
                },
                "DAT": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "fitter_type": "regression",
                },
                "SUBMARKET_ID": {
                    "column_type": "covariate",
                    "value_type": "categorical",
                    "applied_metrics": ["ASAP"],
                },
                "UNIT_ID": {"column_type": "cluster", "applied_metrics": ["ASAP"]},
            },
            "experiment_settings": {"type": "ab"},
        }
        return config_with_customized_covar

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        in_str = """SP_ID,UNIT_ID,ASAP,GROUP1,DAT,FLF,SUBMARKET_ID,GROUP2
        16,846_16,2229,treatment,785,0.514541384,4,1
        16,846_16,1896,treatment,1860,0.48411215,4,1
        16,846_16,992,treatment,963,0.586956522,4,0
        16,846_16,1165,treatment,1137,0.45709282,4,0
        16,846_16,1563,treatment,753,0.522123896,4,0
        16,846_16,3553,treatment,2990,0.443037975,4,0
        16,846_16,1693,treatment,925,0.59,4,1
        16,846_16,1660,treatment,1014,0.459130435,4,0
        16,846_17,1280,control,1169,0.496567501,4,1
        16,846_17,2153,control,1891,0.587646077,4,0
        16,846_17,2181,control,1914,0.583193277,4,0
        16,846_17,1479,control,932,0.522123896,4,1
        16,846_17,1490,control,1442,0.576728499,4,1
        16,846_17,1171,control,947,0.59,4,1
        16,846_17,1339,control,1122,0.429590018,4,0
        16,846_17,2550,control,2465,0.617495712,4,1
        20,846_20,820,control,615,0.71957672,4,1
        20,846_20,2803,control,1871,0.706185567,4,0
        20,846_20,1717,control,1473,0.840000019,4,0
        20,846_20,2449,control,1915,0.804347826,4,0
        20,846_20,3302,control,2475,0.794392523,4,0
        20,846_20,1907,control,1744,0.928229665,4,0
        20,846_20,3157,control,2902,0.820276498,4,0
        20,846_20,1383,control,1336,1.068965536,4,0
        20,846_21,3641,treatment,2423,0.8,4,1
        20,846_21,2104,treatment,2028,0.71957672,4,1
        20,846_21,3191,treatment,2497,0.965116285,4,1
        20,846_21,1917,treatment,1870,0.806818163,4,0
        20,846_21,3213,treatment,2725,0.82464455,4,0
        20,846_21,1225,treatment,885,0.867816107,4,1
        20,846_20,1710,control,1678,0.895953768,4,1
        20,846_20,2304,control,1474,0.813852814,4,0
        401,846_401,2117,control,1061,0.854545455,50,0
        401,846_401,1739,control,1300,0.84516129,50,0
        401,846_401,2657,control,1078,1.345794393,50,1
        401,846_402,2458,treatment,890,1.044776119,50,0
        401,846_402,2310,treatment,1737,1.577981651,50,1
        401,846_402,1706,treatment,1283,0.844311377,50,0
        401,846_402,2914,treatment,1916,1.068181818,50,1
        401,846_402,3319,treatment,2266,0.833333333,50,1
        401,846_402,2487,treatment,1561,1.80952381,50,0
        401,846_402,2069,treatment,1014,0.911764706,50,0
        401,846_401,1796,control,1605,0.833333333,50,0
        401,846_401,1967,control,1469,0.833333333,50,1
        401,846_401,2523,control,1319,0.911764706,50,0
        401,846_401,2719,control,1784,1.729411765,50,1
        401,846_402,1633,treatment,1543,1.029850746,50,0
        401,846_402,2918,treatment,2718,0.948148148,50,1
        401,846_402,4439,treatment,2660,1.345794393,50,0
        401,846_402,1504,treatment,1393,1.638297872,50,1
        401,846_402,3041,treatment,2962,0.897058824,50,1
        401,846_402,2356,treatment,1188,1.411764706,50,1
        2109,846_2109,1824,control,1172,0.845070411,50,0
        2109,846_2109,1253,control,1152,0.745454546,50,0
        2109,846_2109,2255,control,1121,0.876712341,50,0
        2109,846_2109,1828,control,1592,0.714285714,50,1
        2109,846_2108,2386,treatment,1504,0.5,50,0
        2109,846_2108,2646,treatment,1437,0.333333349,50,1
        2109,846_2108,3536,treatment,2540,0.5,50,1
        2109,846_2108,1574,treatment,1043,0.7875,50,0
        """
        return pd.read_csv(StringIO(in_str))

    @pytest.fixture
    def data_with_missing_value(self) -> pd.DataFrame:
        """
        data missing counts:
        group                 2
        group_error           2
        time_index            0
        ar_error              2
        is_treatment_group    2
        order                 0
        asap                  2
        """
        in_str = """
        ,group,group_error,time_index,ar_error,is_treatment_group,order,asap
        0,,8.035644871620853,0,-1.2768483677979405,1.0,0,8.567503306158965
        1,,23.05220302081306,0,0.21660765831479725,0.0,0,21.942654313593085
        2,3.0,8.035644871620853,1,0.3542910384213853,,0,8.647611923844076
        3,6.0,23.05220302081306,1,0.04849568959189762,,0,23.195570631341752
        4,7.0,8.035644871620853,2,-0.8249629226795365,1.0,0,
        5,6.0,23.05220302081306,2,-0.7525144786762085,0.0,0,
        6,4.0,8.035644871620853,3,,1.0,0,7.70402187975307
        7,4.0,23.05220302081306,3,,0.0,0,23.668611155670423
        8,4.0,8.035644871620853,4,3.0409841565697384,1.0,0,6.8329607366863385
        9,2.0,23.05220302081306,4,-2.124473460080633,0.0,0,22.26455282395458
        10,7.0,8.035644871620853,5,1.7164628037090814,1.0,0,9.211546412801912
        11,8.0,23.05220302081306,5,-3.336335313058507,0.0,0,24.16223714125734
        12,6.0,8.035644871620853,6,2.083386860624409,1.0,0,8.063907821193768
        13,6.0,23.05220302081306,6,-1.1252276816790272,0.0,0,23.105768773880435
        14,3.0,8.035644871620853,7,-0.04885350691887469,1.0,0,8.903218065480502
        15,8.0,23.05220302081306,7,-2.363463793985014,0.0,0,22.650451581580356
        16,3.0,,8,-0.4515804427406207,1.0,0,7.684308445105344
        17,4.0,,8,-1.662044679397885,0.0,0,23.545789255511636
        18,3.0,8.035644871620853,9,1.2921667998382538,1.0,0,10.017821788162902
        19,6.0,23.05220302081306,9,-0.8669203963611711,0.0,0,24.248675979966748
        """
        return pd.read_csv(StringIO(in_str))

    @pytest.fixture
    def ab_test_data(self):
        return data_generator.get_ab_test_data()

    @pytest.fixture
    def causal_test_data(self):
        return data_generator.get_causal_test_data()

    @pytest.fixture
    def ab_test_config(self):
        config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                    "variations_split": [0.5, 0.5],
                },
                "asap": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "fitter_type": "regression",
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "fitter_type": "basic",
                    },
                ],
                "dat": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "fitter_type": "regression",
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "fitter_type": "basic",
                    },
                ],
                "lateness_20_min": [
                    {
                        "column_type": "metric",
                        "metric_type": "proportional",
                        "fitter_type": "regression",
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "proportional",
                        "fitter_type": "basic",
                    },
                ],
                "bucket_key": {"column_type": "cluster"},
            },
            "experiment_settings": {"type": "ab"},
        }
        return config

    @pytest.fixture
    def ab_test_sequential_config(self):
        config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                    "variations_split": [0.5, 0.5],
                },
                "asap": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "fitter_type": "basic",
                        Constants.COLUMNS_METRIC_SEQUENTIAL_TESTING_TAU: 1.2,
                    },
                ],
                "bucket_key": {"column_type": "cluster"},
            },
            "experiment_settings": {"type": "ab"},
        }
        return config

    @pytest.fixture
    def data_wo_control(self) -> pd.DataFrame:
        in_str = """SP_ID,UNIT_ID,ASAP,GROUP1,DAT,FLF,SUBMARKET_ID,GROUP2
        16,846_16,2229,treatment,785,0.514541384,4,1
        16,846_16,1896,treatment,1860,0.48411215,4,1
        16,846_16,992,treatment,963,0.586956522,4,0
        16,846_16,1165,treatment,1137,0.45709282,4,0
        16,846_16,1563,treatment,753,0.522123896,4,0
        16,846_16,3553,treatment,2990,0.443037975,4,0
        16,846_16,1693,treatment,925,0.59,4,1
        16,846_16,1660,treatment,1014,0.459130435,4,0
        20,846_20,3641,treatment,2423,0.8,4,1
        20,846_20,2104,treatment,2028,0.71957672,4,1
        20,846_20,3191,treatment,2497,0.965116285,4,1
        20,846_20,1917,treatment,1870,0.806818163,4,0
        20,846_20,3213,treatment,2725,0.82464455,4,0
        20,846_20,1225,treatment,885,0.867816107,4,1
        401,846_401,2458,treatment,890,1.044776119,50,0
        401,846_401,2310,treatment,1737,1.577981651,50,1
        401,846_401,1706,treatment,1283,0.844311377,50,0
        401,846_401,2914,treatment,1916,1.068181818,50,1
        401,846_401,3319,treatment,2266,0.833333333,50,1
        401,846_401,2487,treatment,1561,1.80952381,50,0
        401,846_401,2069,treatment,1014,0.911764706,50,0
        401,846_401,1633,treatment,1543,1.029850746,50,0
        401,846_401,2918,treatment,2718,0.948148148,50,1
        401,846_401,4439,treatment,2660,1.345794393,50,0
        401,846_401,1504,treatment,1393,1.638297872,50,1
        401,846_401,3041,treatment,2962,0.897058824,50,1
        401,846_401,2356,treatment,1188,1.411764706,50,1
        2109,846_2109,2386,treatment,1504,0.5,50,0
        2109,846_2109,2646,treatment,1437,0.333333349,50,1
        2109,846_2109,3536,treatment,2540,0.5,50,1
        2109,846_2109,1574,treatment,1043,0.7875,50,0
        """
        return pd.read_csv(StringIO(in_str))
