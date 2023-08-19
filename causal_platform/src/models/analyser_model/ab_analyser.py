"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import sys
import traceback
from typing import List

from causal_platform.src.models.analyser_model.analyser import BaseAnalyser
from causal_platform.src.models.analyser_model.fitters.basic_fitter import BasicFitter
from causal_platform.src.models.analyser_model.fitters.bootstrap_fitter import (
    QuantileBootstrapFitter,
    RatioBootstrapFitter,
)
from causal_platform.src.models.analyser_model.fitters.fitter import SMFFitter
from causal_platform.src.models.configuration_model.base_objects import FitterType, Metric, MetricType
from causal_platform.src.models.configuration_model.config import AbConfig
from causal_platform.src.models.message.enums import Source, Status
from causal_platform.src.models.message.message import Message
from causal_platform.src.models.result_model.result import AbPreprocessPipelineResult, AnalysisResult, CovariateResult
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.error import InputConfigError


class AbAnalyser(BaseAnalyser):
    def __init__(self, config: AbConfig, preprocess_result: AbPreprocessPipelineResult):
        super().__init__(config, preprocess_result)

    def _get_fitter(self, metric):
        if metric.fitter_type == FitterType.basic:
            fitter = BasicFitter(
                self.preprocess_result.processed_data,
                metric,
                self.config.experiment_groups,
                metric.cluster,
                covariates=metric.covariates,
                interaction=self.config.interaction,
                use_iterative_cv_method=self.config.use_iterative_cv_method,
                information_rates=self.config.information_rates,
                target_sample_size=self.config.target_sample_size,
                current_sample_size=self.config.current_sample_size,
            )
        elif metric.fitter_type == FitterType.bootstrap:
            raise InputConfigError(
                "'bootstrap' fitter doesn't support mean metric, please use 'basic' or 'regression' fitter."
            )

        elif metric.fitter_type == FitterType.regression:
            fitter = SMFFitter(
                data=self.preprocess_result.processed_data,
                metric=metric,
                experiment_groups=self.config.experiment_groups,
                covariates=metric.covariates,
                cluster=metric.cluster,
                fixed_effect_estimator=self.config.fixed_effect_estimator,
                interaction=self.config.interaction,
            )
        else:
            raise InputConfigError("causal-platform only support 'regression', 'basic' and 'bootstrap' fitter type.")

        return fitter

    def _get_quantile_fitter(self, metric):
        if metric.fitter_type == FitterType.bootstrap:
            return QuantileBootstrapFitter(
                data=self.preprocess_result.processed_data,
                metric=metric,
                experiment_groups=self.config.experiment_groups,
                cluster=metric.cluster,
                bootstrap_size=self.config.bootstrap_size,
                iteration=self.config.bootstrap_iteration,
            )
        elif metric.fitter_type == FitterType.basic:
            return BasicFitter(
                data=self.preprocess_result.processed_data,
                metric=metric,
                experiment_groups=self.config.experiment_groups,
                cluster=metric.cluster,
                interaction=self.config.interaction,
                use_iterative_cv_method=self.config.use_iterative_cv_method,
            )
        else:
            raise InputConfigError(
                "causal-platform only support 'basic' and 'bootstrap' fitter type for quantile metrics."
            )

    def _get_ratio_fitter(self, metric):
        if metric.fitter_type == FitterType.bootstrap:
            return RatioBootstrapFitter(
                data=self.preprocess_result.processed_data,
                metric=metric,
                experiment_groups=self.config.experiment_groups,
                cluster=metric.cluster,
                bootstrap_size=self.config.bootstrap_size,
                iteration=self.config.bootstrap_iteration,
            )
        elif metric.fitter_type == FitterType.basic:
            return BasicFitter(
                data=self.preprocess_result.processed_data,
                metric=metric,
                experiment_groups=self.config.experiment_groups,
                interaction=self.config.interaction,
                covariates=metric.covariates,
                use_iterative_cv_method=self.config.use_iterative_cv_method,
            )
        else:
            raise InputConfigError(
                "causal-platform only support 'bootstrap' and 'basic' fitter type for ratio metrics."
            )

    def _analyze_metric(self, metric: Metric) -> List[AnalysisResult]:
        if metric.metric_type == MetricType.ratio:
            fitter = self._get_ratio_fitter(metric)
        elif metric.metric_aggregate_func.name == Constants.STATISTICS_CALCULATE_FUNC_QUANTILE:
            fitter = self._get_quantile_fitter(metric)
        else:
            fitter = self._get_fitter(metric)
        self.fitter_dict[metric.column_name] = fitter
        fitter.group_sequential = self.group_sequential
        fitter.fit()

        analysis_results = fitter.get_analysis_results()
        self.message_collection.combine(fitter.message_collection)
        return analysis_results

    def get_covariate_results(self) -> List[CovariateResult]:
        results = []
        for metric in self.config.metrics:
            try:
                fitter = self.fitter_dict[metric.column_name]
                results.extend(fitter.get_covariate_results())
            except Exception as e:
                # Rip out the string representation that would be formed by print_exception()
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exceptions = traceback.format_exception(exc_type, exc_value, exc_traceback)
                # Add a catchall message
                self.message_collection.log_message(
                    Message(
                        source=Source.analysis,
                        status=Status.fail,
                        title=f"Unable to get covariates inference for metric {metric.column_name}",
                        description=f"""Unable to get covariates inference for metric {metric.column_name} due to {e}, will skip: {"".join(exceptions)}""",
                    ),
                )
        return results
