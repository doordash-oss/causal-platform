"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import datetime
from typing import List, Optional, Set, TypeVar

import numpy as np

from causal_platform.src.models.configuration_model.base_objects import (
    CheckImbalanceMethod,
    Cluster,
    Column,
    ColumnType,
    Covariate,
    CovariateType,
    DateColumn,
    ExperimentGroup,
    ExperimentType,
    MatchingMethod,
    Metric,
    MetricType,
    TColumn,
    FitterType,
)
from causal_platform.src.utils.error import InputConfigError
from causal_platform.src.utils.logger import logger


class PipelineConfig:
    def __init__(self, copy_input_data=True):
        """
        :param copy_input_data: If True a copy of the DataFrame is made. If False the input DataFrame will likely be
        modified in place during validation and transformation
        """
        self.copy_input_data = copy_input_data


class BaseConfig:
    def __init__(
        self,
        covariates: Optional[List[Covariate]],
        metrics: List[Metric],
        experiment_type: ExperimentType,
        experiment_randomize_units: Optional[List[Column]],
        clusters: Optional[List[Cluster]] = None,
    ) -> None:

        from causal_platform.src.utils.config_utils import set_metrics

        self.covariates = covariates if covariates is not None else []
        self.clusters = clusters if clusters is not None else []
        self.metrics = metrics
        set_metrics(self.metrics, self.clusters, self.covariates)
        self.experiment_type = experiment_type
        self.experiment_randomize_units = experiment_randomize_units if experiment_randomize_units is not None else []
        self._validate_config()

    @property
    def columns(self):
        covariates = set()
        clusters = set()
        for metric in self.metrics:
            covariates.update(metric.covariates)
            clusters.add(metric.cluster)

        columns: List[TColumn] = self.metrics + list(covariates) + list(clusters)

        return columns

    # for ab, experiment_randomize_unit is the unit that you randomize on
    # i.e dasher_id, delivery_id, etc
    # for diff-in-diff, experiment unit is market/region/store/...
    @property
    def cluster(self):
        return self.clusters[0] if len(self.clusters) > 0 else None

    @property
    def all_distinct_covariates(self):
        covariates = set()
        for metric in self.metrics:
            if metric.covariates:
                covariates.update(metric.covariates)
        return list(covariates)

    @property
    def all_distinct_clusters(self):
        clusters = set()
        for metric in self.metrics:
            if metric.cluster:
                clusters.add(metric.cluster)
        return list(clusters)

    def get_metrics_by_metric_name(self, metric_name: str) -> List[Metric]:
        metrics = []
        for metric in self.metrics:
            if metric.column_name == metric_name:
                metrics.append(metric)
        return metrics

    def get_metric_type(self, metric_name: str) -> MetricType:
        for metric in self.metrics:
            if metric.column_name == metric_name:
                return metric.metric_type

    def get_columns_by_column_type(self, column_type: ColumnType) -> List[TColumn]:
        # TODO: implement the function to get columns by the column type
        column = Column("temp", ColumnType.metric)
        return [column]

    def _validate_config(self):
        # Check there is at least one metric
        if len(self.metrics) == 0:
            logger.warning("No metrics passed into the analysis, conduct flicker and imbalance check only!")
        # Check whether config settings is correct
        metric_names = [metric.column_name for metric in self.metrics]
        for covariate in self.covariates:
            if covariate.column_name in metric_names:
                raise InputConfigError(f"Metrics Column {covariate.column_name} can not be used as covariates")
            applied_metric_names = covariate.applied_metric_names

            for metric_name in applied_metric_names:
                if metric_name not in metric_names:
                    raise InputConfigError(
                        f"covariate {covariate.column_name} has an unrecognized metric name that is trying to apply: {metric_name}"
                    )

        for metric in self.metrics:
            covariates = metric.covariates
            if metric.metric_type == MetricType.ratio:
                if metric.cluster is not None:
                    raise InputConfigError(
                        f"{metric.column_name} is a ratio metric that has cluster {metric.cluster.column_name} which is not supported"
                    )
                for covariate in covariates:
                    if covariate.value_type != CovariateType.ratio:
                        raise InputConfigError(
                            f"{metric.column_name} is a ratio metric that has covariate {covariate.column_name} that is not also ratio type"
                        )
            else:  # non-ratio case
                for covariate in covariates:
                    if covariate.value_type == CovariateType.ratio:
                        raise InputConfigError(
                            f"{metric.column_name} is not a ratio metric but has a covariate {covariate.column_name} that is a ratio type"
                        )

    # TODO: check contains at least one metric column, one treatament group columns
    # TODO: interaction between metrics at preprocessing

    def get_required_columns_names(self) -> Set[str]:
        column_names = set(
            column.column_name
            for column in self.all_distinct_covariates + self.all_distinct_clusters + self.experiment_randomize_units
        )

        for metric in self.metrics:
            if metric.metric_type == MetricType.ratio:
                column_names.update([metric.denominator_column.column_name, metric.numerator_column.column_name])
            else:
                column_names.add(metric.column_name)

        for covariate in self.covariates:
            if covariate.value_type == CovariateType.ratio:
                column_names.update(
                    [
                        covariate.denominator_column.column_name,
                        covariate.numerator_column.column_name,
                    ]
                )
            else:
                column_names.add(covariate.column_name)

        return column_names


class AbConfig(BaseConfig):
    def __init__(
        self,
        experiment_groups: List[ExperimentGroup],
        experiment_randomize_units: Optional[List[Column]],
        metrics: List[Metric],
        is_check_flickers: bool = True,
        is_remove_flickers: bool = False,
        is_check_imbalance: bool = True,
        check_imbalance_method: CheckImbalanceMethod = CheckImbalanceMethod.chi_square,
        is_check_metric_type: bool = False,
        date: Optional[DateColumn] = None,
        covariates: Optional[List[Covariate]] = None,
        experiment_type: ExperimentType = ExperimentType.ab,
        clusters: Optional[List[Column]] = None,
        bootstrap_size: Optional[int] = None,
        bootstrap_iteration: Optional[int] = None,
        fixed_effect_estimator: bool = False,
        interaction: bool = False,
        is_simulation: bool = False,
        use_iterative_cv_method: bool = True,
        information_rates: Optional[List[float]] = None,
        target_sample_size: Optional[int] = None,
        current_sample_size: Optional[int] = None,
    ):
        super().__init__(
            covariates,
            metrics,
            experiment_type=experiment_type,
            experiment_randomize_units=experiment_randomize_units,
            clusters=clusters,
        )
        self.date = date
        self.experiment_groups = experiment_groups
        self.is_remove_flickers = is_remove_flickers
        self.is_check_imbalance = is_check_imbalance
        self.check_imbalance_method = check_imbalance_method
        self.is_check_flickers = is_check_flickers or is_remove_flickers
        self.is_check_metric_type = is_check_metric_type
        self.bootstrap_size = bootstrap_size
        self.bootstrap_iteration = bootstrap_iteration
        self.fixed_effect_estimator = fixed_effect_estimator
        self.interaction = interaction
        self.is_simulation = is_simulation
        self.use_iterative_cv_method = use_iterative_cv_method
        self.information_rates = information_rates
        self.target_sample_size = target_sample_size
        self.current_sample_size = current_sample_size
        self._validate_ab_config()

    @property
    def columns(self):
        columns = super().columns
        columns.extend(self.experiment_groups)
        return columns

    def _validate_ab_config(self):
        # experiment_group can't be empty
        if (
            not self.is_simulation
            and len(self.experiment_groups) == 0
            and (self.experiment_type != ExperimentType.causal)
        ):
            raise InputConfigError("'experiment_group' is not provided in the config")

        # covariate can't be used as metric
        group_date_column_names = []
        if self.date is not None:
            group_date_column_names.append(self.date)
        group_date_column_names.extend(self.experiment_groups)
        for covariate in self.covariates:
            if covariate.column_name in group_date_column_names:
                raise InputConfigError(
                    f"Experiment group/Date Column {covariate.column_name} can not be used as covariates"
                )
        # when there is interaction, only allow two experiment_group
        if self.interaction:
            if len(self.experiment_groups) > 2:
                raise InputConfigError(
                    "causal-platform only support two way interaction of two experiment groups! There are more than two experiment_groups in the config."
                )
            if len(self.experiment_groups) == 1:
                raise InputConfigError(
                    "There is only one experiment_group but interaction is set to True. There must be two experiment_groups for interaction analysis."
                )

        if self.information_rates:
            if min(self.information_rates) <= 0 or max(self.information_rates) > 1.0:
                raise InputConfigError("Information rates should be bounded between between (0, 1]")
            if not np.all(np.diff(self.information_rates) > 0):
                raise InputConfigError(
                    "Information rates should be monotonically increasing (e.g., [0.1, 0.2, 0.8, 1.0]"
                )
            if len(self.information_rates) != len(set(self.information_rates)):
                raise InputConfigError("Information rates should not contain duplications")

        if self.experiment_type == ExperimentType.causal:
            for metric in self.metrics:
                if metric.fitter_type != FitterType.basic:
                    raise InputConfigError("We only support basic fitter for causal analysis")
                for covariate in metric.covariates:
                    if covariate.value_type != CovariateType.ratio or metric.metric_type != MetricType.ratio:
                        # have this because causal analysis is only built for ratio type now
                        raise InputConfigError("All metric and covariates need to be ratio for causal analysis")

    def get_required_columns_names(self) -> Set[str]:
        column_names = super().get_required_columns_names()
        if self.date:
            column_names.add(self.date.column_name)
        column_names.update([column.column_name for column in self.experiment_groups])
        return column_names


class DiDConfig(BaseConfig):
    def __init__(
        self,
        metrics: List[Metric],
        experiment_randomize_units: List[Column],
        date: DateColumn,
        treatment_unit_ids: List[int],
        exclude_unit_ids: Optional[List[int]],
        match_unit_size: int,
        matching_start_date: datetime.datetime,
        matching_end_date: datetime.datetime,
        experiment_start_date: datetime.datetime,
        experiment_end_date: datetime.datetime,
        matching_columns: List[str],
        matching_weights: List[float],
        covariates: Optional[List[Covariate]] = None,
        matching_method: MatchingMethod = MatchingMethod.correlation,
        experiment_type: ExperimentType = ExperimentType.diff_in_diff,
        small_sample_adjustment: bool = True,
    ):
        first_randomize_unit = experiment_randomize_units[0]
        cluster = Cluster(
            column_name=first_randomize_unit.column_name,
        )
        super().__init__(
            covariates,
            metrics,
            experiment_type=experiment_type,
            experiment_randomize_units=experiment_randomize_units,
            clusters=[cluster],
        )
        self.treatment_unit_ids = treatment_unit_ids
        self.match_unit_size = match_unit_size
        self.matching_method = matching_method
        self.matching_start_date = matching_start_date
        self.matching_end_date = matching_end_date
        self.experiment_start_date = experiment_start_date
        self.experiment_end_date = experiment_end_date
        self.date = date
        if exclude_unit_ids is None:
            self.exclude_unit_ids = []
        else:
            self.exclude_unit_ids = exclude_unit_ids
        self.matching_columns = matching_columns
        self.matching_weights = matching_weights
        self.small_sample_adjustment = small_sample_adjustment
        self.validate_diff_in_diff_config()

    def get_diff_in_diff_experiment_unit_column(self) -> Column:
        # Take the first value as the unit column
        return self.experiment_randomize_units[0]

    def validate_diff_in_diff_config(self):
        # validate that the experiment type is diff in diff
        if self.experiment_type != ExperimentType.diff_in_diff:
            raise InputConfigError(
                f"Experiment Type is not '{ExperimentType.diff_in_diff}'. Found '{self.experiment_type.value}'"
            )

        # validate that the matching weights sum up to 1
        if round(sum(self.matching_weights), 2) != 1:
            raise InputConfigError(f"Matching weights do not sum to 1. Found '{self.matching_weights}'")

        # validate that matching weights are non-negative
        if min(self.matching_weights) < 0:
            raise InputConfigError(f"Matching weights cannot be negative. Found '{min(self.matching_weights)}' value")

        # validate matching method is valid
        if not MatchingMethod.has_value(self.matching_method.value):
            raise InputConfigError(f"Matching Method '{self.matching_method.value}' is not valid")

        # validate that the randomize unit column exists
        if not self.experiment_randomize_units:
            raise InputConfigError("Randomize Unit column does not exist")

        # validate that randomize unit column is the expected type
        for column in self.experiment_randomize_units:
            if column.column_type != ColumnType.experiment_randomize_unit:
                raise InputConfigError(
                    f"Randomize Unit Column '{column.column_name}' has invalid ColumnType '{column.column_type}'"
                )

        # validate that the randomize unit column exists
        if len(self.experiment_randomize_units) != 1:
            err_msg = """"Randomize Unit column has length '{}'.
             Currently only one randomize unit column supported."""
            raise InputConfigError(err_msg.format(len(self.experiment_randomize_units)))

        # validate that there is only on metric specific
        # currently diff in diff only supports one metric
        if len(self.metrics) != 1:
            raise InputConfigError(f"Diff in Diff only supports 1 metric, '{len(self.metrics)}' found")

    def get_required_columns_names(self) -> Set[str]:
        column_names = super().get_required_columns_names()
        column_names.add(self.date.column_name)
        column_names.update([column_name for column_name in self.matching_columns])
        return column_names


TBaseConfig = TypeVar("TBaseConfig", bound=BaseConfig)


class PowerCalculatorConfig:
    def __init__(
        self,
        covariates: Optional[List[Covariate]],
        metrics: List[Metric],
        clusters: Optional[List[Cluster]] = None,
        experiment_type: ExperimentType = ExperimentType.ab,
    ) -> None:
        from causal_platform.src.utils.config_utils import set_metrics

        self.covariates = covariates if covariates is not None else []
        self.clusters = clusters if clusters is not None else []
        self.metrics = metrics
        self.experiment_type = experiment_type
        set_metrics(self.metrics, self.clusters, self.covariates)

    @property
    def columns(self):
        # Prepare columns to remove NaN
        covariates = set()
        clusters = set()
        for metric in self.metrics:
            covariates.update(metric.covariates)
            clusters.add(metric.cluster)

        columns: List[TColumn] = self.metrics + list(covariates) + list(clusters)

        return columns
