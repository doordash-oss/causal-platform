"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from causal_platform.src.models.configuration_model.base_objects import (
    ExperimentGroup,
    ExperimentVariation,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.models.result_model.result import AnalysisResult


class TestResult:
    def test_validate(self):
        is_valid = self.build_result().validate()
        assert is_valid

    def test_estimated_treatment_effect_validate(self):
        is_valid = self.build_result(estimated_treatment_effect=float("nan")).validate()
        assert not is_valid

    def test_p_value_validate(self):
        is_valid = self.build_result(p_value=float("nan")).validate()
        assert not is_valid

    def test_confidence_value_validate(self):
        is_valid = self.build_result(confidence_interval_left=float("nan")).validate()
        assert not is_valid

        is_valid = self.build_result(confidence_interval_right=float("nan")).validate()
        assert not is_valid

    def test_se_validate(self):
        is_valid = self.build_result(se=float("nan")).validate()
        assert not is_valid

    def test_metric_treatment_value_validate(self):
        is_valid = self.build_result(metric_treatment_value=float("nan")).validate()
        assert not is_valid

    def test_metric_control_value_validate(self):
        is_valid = self.build_result(metric_control_value=float("nan")).validate()
        assert not is_valid

    def test_sequential_p_value_validate(self):
        is_valid = self.build_result(sequential_p_value=float("nan")).validate()
        assert not is_valid

    def test_sequential_confidence_value_validate(self):
        is_valid = self.build_result(sequential_confidence_interval_left=float("nan")).validate()
        assert not is_valid

        is_valid = self.build_result(sequential_confidence_interval_right=float("nan")).validate()
        assert not is_valid

    def build_result(
        self,
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
