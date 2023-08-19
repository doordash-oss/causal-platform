"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from causal_platform.src.models.analyser_model.fitters.basic_fitter import BasicFitter
from causal_platform.src.models.configuration_model.base_objects import ExperimentGroup
from causal_platform.src.models.configuration_model.config import TBaseConfig
from causal_platform.src.utils.common_utils import (
    convert_data_to_proper_types,
    convert_table_column_to_lower_case,
)
from causal_platform.src.utils.config_utils import set_power_calculator_config
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.error import InputConfigError, InputDataError
from causal_platform.src.utils.experiment.result_utils import calculate_basic_sample_metric_stats
from causal_platform.src.utils.logger import logger


class ABPowerCalculator:
    def __init__(
        self,
        data: pd.DataFrame,
        config: Union[Dict, TBaseConfig],
        logger_type: Optional[str] = None,
    ):
        logger.reset_logger_with_type(logger_type)
        self.config = set_power_calculator_config(config)
        self.data = data
        self._clean_data_for_power_calculator()

    def _clean_data_for_power_calculator(self):
        convert_table_column_to_lower_case(self.data)
        convert_data_to_proper_types(self.data, self.config.metrics, covariates=self.config.covariates)

    def _calculate_power(self, metric, standard_deviation):
        effect_size = self._calculate_effect_size(metric.absolute_treatment_effect, standard_deviation)
        n = metric.sample_size_per_group
        threshold_1 = np.sqrt(n / 2.0) * np.abs(effect_size) - norm.ppf(1 - metric.alpha / 2)
        threshold_2 = -1 * np.sqrt(n / 2.0) * effect_size - norm.ppf(1 - metric.alpha / 2)
        return norm.cdf(threshold_1) + norm.cdf(threshold_2)

    def _calculate_sample_size(self, metric, standard_deviation):
        """
        calculate sample size required to get certain power defined in metric
        if the data is clustered, result is the number of clusters rather than samples
        """
        if metric.power == 0:
            return 0
        effect_size = self._calculate_effect_size(metric.absolute_treatment_effect, standard_deviation)
        sample_size_required = 2 * (self._get_multiplier(metric) / effect_size) ** 2
        return sample_size_required

    def _get_multiplier(self, metric):
        """
        The sum of ppf of 1 - alpha/2 and ppf of 1 - beta (power), which is
        the ratio between minimum detectable difference and s.e.
        """
        return norm.ppf(1 - metric.alpha / 2) + norm.ppf(metric.power)

    def _calculate_mde(self, metric, standard_deviation):
        n = metric.sample_size_per_group
        required_effect_size = self._get_multiplier(metric) / np.sqrt(n / 2.0)
        mde = required_effect_size * standard_deviation
        return mde

    def _calculate_effect_size(self, mde, standard_deviation):
        """
        the number of standard deviations between MDE
        """
        return mde / standard_deviation

    def _calculate_standard_deviation(self, metric):
        """
        the standard deviation (sigma) of the population
        """
        self.fitter = BasicFitter(
            data=self.data,
            metric=metric,
            cluster=metric.cluster,
            experiment_groups=[ExperimentGroup("FakeGroup")],
            covariates=metric.covariates,
        )
        self.fitter.fit()

        sample_mean_variance = self.fitter.compute_sample_statistics_variance(self.fitter.data)
        n = (
            self.fitter.data.shape[0]
            if self.fitter.cluster is None
            else self.fitter.data[self.fitter.cluster.column_name].nunique()
        )
        return np.sqrt(sample_mean_variance * n)

    def run_metric_stats_calculator(self):
        """
        this is the endpoint for curie to calculate and cache metric stats for metrics.
        """
        metric_stats = {}
        for metric in self.config.metrics:
            try:
                standard_deviation = self._calculate_standard_deviation(metric)
                metric_value, sample_size, data_size = calculate_basic_sample_metric_stats(self.data, metric)
                metric_stats[metric.column_name] = {
                    Constants.POWER_CALCULATOR_STANDARD_DEVIATION: standard_deviation,
                    Constants.POWER_CALCULATOR_METRIC_VALUE: metric_value,
                    Constants.POWER_CALCULATOR_SAMPLE_SIZE: sample_size,
                    Constants.POWER_CALCULATOR_DATA_SIZE: data_size,
                }
            except Exception as e:
                logger.error(f"failed to calculate metric stats for metric {metric} due to {e}")
        return metric_stats

    def run(self):
        sample_size_list = []
        mde_list = []
        power_list = []
        metric_names = []
        for metric in self.config.metrics:
            mde = metric.absolute_treatment_effect
            sample_size_per_group = metric.sample_size_per_group
            power = metric.power
            try:
                method = None
                count_element = sum([power is not None, sample_size_per_group is not None, mde is not None])
                if count_element != 2:
                    raise InputConfigError(
                        f"""two of the three elements (power, sample_size_per_group, absolute_treatment_effect) \
                         need to be provided for every metric! {count_element} are provided for metric {metric.column_name}
                         """
                    )
                elif power is None:
                    method = Constants.POWER
                elif sample_size_per_group is None:
                    method = Constants.SAMPLE_SIZE_PER_GROUP
                elif mde is None:
                    method = Constants.ABS_TREATMENT_EFFECT
                standard_deviation = self._calculate_standard_deviation(metric)
                if method == Constants.POWER:
                    assert mde is not None and sample_size_per_group is not None
                    power_list.append(self._calculate_power(metric, standard_deviation))
                    mde_list.append(mde)
                    sample_size_list.append(sample_size_per_group)
                elif method == Constants.SAMPLE_SIZE_PER_GROUP:
                    assert mde is not None and power is not None
                    sample_size_list.append(self._calculate_sample_size(metric, standard_deviation))
                    mde_list.append(mde)
                    power_list.append(power)
                else:
                    assert sample_size_per_group is not None and power is not None
                    mde_list.append(self._calculate_mde(metric, standard_deviation))
                    sample_size_list.append(sample_size_per_group)
                    power_list.append(power)
                metric_names.append(metric.column_name)
            except Exception as e:
                logger.error(f"unable to analyze metric {metric} due to {e}")
        if len(metric_names) == 0:
            raise InputDataError("failed to run calculator for all metrics!")
        return pd.DataFrame(
            list(zip(metric_names, sample_size_list, mde_list, power_list)),
            columns=[
                Constants.OUTPUT_METRIC,
                Constants.SAMPLE_SIZE_PER_GROUP,
                Constants.ABS_TREATMENT_EFFECT,
                Constants.POWER,
            ],
        )
