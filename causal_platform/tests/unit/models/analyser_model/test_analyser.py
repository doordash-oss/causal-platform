"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import List

import pandas as pd

from causal_platform.src.models.analyser_model.analyser import BaseAnalyser
from causal_platform.src.models.configuration_model.base_objects import (
    ExperimentGroup,
    ExperimentType,
    ExperimentVariation,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.models.configuration_model.config import BaseConfig
from causal_platform.src.models.result_model.result import AnalysisResult, BasePreprocessPipelineResult


class TestAnalyser:
    def test_valid_results(self):
        fake_analysis_results = [build_result(), build_result(), build_result()]

        analyser = FakeAnalyser(fake_analysis_results)
        results = analyser.run()
        assert len(results) == len(fake_analysis_results), "Expected a result for each valid fake result"
        assert len(analyser.message_collection.overall_messages) == 0, "Expected no overall errors"
        assert len(analyser.message_collection.metric_messages) == 0, "Expected no metric errors"

    def test_invalid_results(self):
        fake_analysis_results = [
            build_result(p_value=float("nan")),
            build_result(),
            build_result(),
        ]

        analyser = FakeAnalyser(fake_analysis_results)
        results = analyser.run()
        assert len(results) == 2, "Expected 2 valid results and 1 invalid"
        assert len(analyser.message_collection.overall_messages) == 0, "Expected no overall errors"
        assert len(analyser.message_collection.metric_messages) == 1, "Expected no metric errors"


# Fake Analyser for testing ----------------------------------------------------------
class FakeAnalyser(BaseAnalyser):
    def __init__(self, results: List[AnalysisResult]):
        super().__init__(config=create_config(results), preprocess_result=create_preprocess_result())

        self.results = results
        self.count = 0

    def _analyze_metric(self, metric) -> List[AnalysisResult]:
        result = [self.results[self.count]]
        self.count += 1
        return result


# Builder functions ----------------------------------------------------------------
def create_config(results: List[AnalysisResult]) -> BaseConfig:
    metrics = []
    for _ in results:
        metrics.append(
            Metric(
                column_name="column",
                metric_type=MetricType.continuous,
                metric_aggregate_func=MetricAggregateFunc.mean,
            )
        )

    return BaseConfig(
        covariates=None,
        metrics=metrics,
        experiment_type=ExperimentType.ab,
        experiment_randomize_units=None,
        clusters=None,
    )


def create_preprocess_result() -> BasePreprocessPipelineResult:
    return BasePreprocessPipelineResult(processed_data=pd.DataFrame())


def build_result(
    estimated_treatment_effect: float = 0.0,
    p_value: float = 0.0,
    confidence_interval_left: float = 0.0,
    confidence_interval_right: float = 0.0,
    se: float = 0.0,
    metric_treatment_value: float = 0.0,
    metric_control_value: float = 0.0,
    sequential_p_value: float = 0.0,
    sequential_confidence_interval_left: float = 0.0,
    sequential_confidence_interval_right: float = 0.0,
) -> AnalysisResult:
    return AnalysisResult(
        metric=Metric(
            column_name="fake",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
        ),
        estimated_treatment_effect=estimated_treatment_effect,
        p_value=p_value,
        confidence_interval_left=confidence_interval_left,
        confidence_interval_right=confidence_interval_right,
        experiment_group=ExperimentGroup(column_name="some_column"),
        experiment_group_variation=ExperimentVariation("treatment", 0.5),
        se=se,
        metric_treatment_value=metric_treatment_value,
        metric_control_value=metric_control_value,
        metric_treatment_sample_size=0,
        metric_control_sample_size=0,
        metric_treatment_data_size=0,
        metric_control_data_size=0,
        is_sequential_result_valid=True,
        sequential_p_value=sequential_p_value,
        sequential_confidence_interval_left=sequential_confidence_interval_left,
        sequential_confidence_interval_right=sequential_confidence_interval_right,
    )
