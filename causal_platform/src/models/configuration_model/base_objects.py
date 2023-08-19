"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from enum import Enum
from typing import List, Optional, TypeVar, Union

import numpy as np

from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.error import InputConfigError
from causal_platform.src.utils.logger import logger


class SuperEnum(Enum):
    @classmethod
    def all_values(cls) -> List[str]:
        return list(map(lambda x: x.value, cls))

    @classmethod
    def has_value(cls, value) -> bool:
        return value in cls.all_values()

    @classmethod
    def _missing_(cls, value):
        # Override exception class of python Enum for causal-platform Config information
        raise InputConfigError("%r is not a valid %s" % (value, cls.__name__))


class SequentialResultType(SuperEnum):
    group = "GROUP_SEQUENTIAL"
    always_valid = "ALWAYS_VALID"


class MetricType(SuperEnum):
    continuous = "continuous"
    proportional = "proportional"
    ratio = "ratio"


class CovariateType(SuperEnum):
    categorial = "categorical"
    numerical = "numerical"
    ratio = "ratio"
    # TODO: potentially alter this to distinguish numerical, categorical ratio


class ExperimentType(SuperEnum):
    ab = "ab"
    diff_in_diff = "diff_in_diff"
    causal = "causal"


class FitterType(SuperEnum):
    basic = "basic"
    regression = "regression"
    bootstrap = "bootstrap"


class MetricAggregateFunc(SuperEnum):
    sum = "sum"
    mean = "mean"
    quantile = "quantile"


class ColumnType(SuperEnum):
    date = "date"
    experiment_group = "experiment_group"
    metric = "metric"
    covariate = "covariate"
    experiment_randomize_unit = "experiment_randomize_unit"
    cluster = "cluster"
    ratio_metric_component = "ratio_metric_component"
    ratio_covariate_component = "ratio_covariate_component"


class MatchingMethod(SuperEnum):
    correlation = "correlation"
    euclidean_distance = "euclidean_distance"


class HypothesisTesting(SuperEnum):
    t_test = "t_test"
    rank_test = "rank_test"


class CheckImbalanceMethod(SuperEnum):
    chi_square = "chi-square"
    binomial = "binomial"


class Column:
    def __init__(self, column_name: str, column_type: ColumnType):
        self.original_column_name = column_name
        self.column_name = str.lower(column_name)
        self.column_type = column_type

    def __str__(self):
        return f"Column {self.column_name}"

    def __repr__(self):
        return f"Column({self._get_object_summary()})"

    def _get_object_summary(self):
        return ",".join([f"{k}={v}" for k, v in self.__dict__.items()])

    def __eq__(self, o: object) -> bool:
        return self.__class__ == o.__class__ and self.__repr__() == o.__repr__()

    def __hash__(self) -> int:
        return hash(self.__repr__())


TColumn = TypeVar("TColumn", bound=Column)


class Covariate(Column):
    def __init__(
        self,
        column_name: str,
        value_type: CovariateType,  # options: categorical, numerical, ratio
        applied_metric_names: Optional[List[str]] = None,
        numerator_column: Optional[Column] = None,
        denominator_column: Optional[Column] = None,
        # the coefficient of ratio covariate, the value is set sequentially during variance reduction
        coef: Optional[float] = None,
    ):
        super().__init__(column_name, ColumnType.covariate)
        self.value_type = value_type
        self.applied_metric_names = [metric.lower() for metric in applied_metric_names] if applied_metric_names else []
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column
        self.coef = coef
        self._validate_all()

    def __str__(self):
        return f"Covariate {self.column_name}"

    def __repr__(self):
        return f"Covariate({self._get_object_summary()})"

    # checks if user provided only one of numerator or denominator
    def _validate_ratio(self):
        if self.value_type == CovariateType.ratio:
            if self.numerator_column is None or self.denominator_column is None:
                raise InputConfigError(
                    f"Please provide numerator and denominator columns for ratio covariate '{self.column_name}' in config."
                )

    def _validate_all(self):
        self._validate_ratio()


class Cluster(Column):
    def __init__(
        self,
        column_name: str,
        applied_metric_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(column_name, ColumnType.cluster)
        self.applied_metric_names = [metric.lower() for metric in applied_metric_names] if applied_metric_names else []

    def __str__(self):
        return f"Cluster {self.column_name}"

    def __repr__(self):
        return f"Cluster({self._get_object_summary()})"


class Metric(Column):
    def __init__(
        self,
        column_name: str,
        metric_type: MetricType,
        metric_aggregate_func: MetricAggregateFunc,
        log_transform: bool = False,
        remove_outlier: bool = False,
        check_distribution: bool = False,
        alpha: float = 0.05,
        quantile: Optional[float] = None,
        numerator_column: Optional[Column] = None,
        denominator_column: Optional[Column] = None,
        applied_covariate_names: Optional[List[str]] = None,
        covariates: Optional[List[Covariate]] = None,
        clusters: Optional[List[Cluster]] = None,
        fitter_type: Optional[FitterType] = FitterType.regression,
        # used for power calculator only
        absolute_treatment_effect: Optional[float] = None,
        power: Optional[float] = None,
        sample_size_per_group: Optional[float] = None,
        sequential_testing_tau: Optional[float] = None,
    ):
        super().__init__(column_name, ColumnType.metric)
        self.metric_type = metric_type
        self.metric_aggregate_func = metric_aggregate_func
        self.quantile = quantile
        self.log_transform = log_transform
        self.remove_outlier = remove_outlier
        self.check_distribution = check_distribution
        self.alpha = alpha
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column
        self.clusters = clusters or []
        self.applied_covariate_names = applied_covariate_names or []
        self.covariates = covariates or []
        self.fitter_type = fitter_type
        self.absolute_treatment_effect = absolute_treatment_effect
        self.power = power
        self.sample_size_per_group = sample_size_per_group
        self.sequential_testing_tau = sequential_testing_tau if sequential_testing_tau != 0 else 1
        self._processed_column_name = None
        self._validate_all()

    @property
    def processed_column_name(self):
        if self._processed_column_name is None:
            return self.column_name
        else:
            return self._processed_column_name

    def set_processed_column_name(self, processed_column_name):
        self._processed_column_name = processed_column_name

    @property
    def cluster(self):
        """
        TODO: now we only support 1 cluster so adding this "cluster" property to take
        the first value in the list. In future when we support multiple cluster,
        can remove this.
        """
        return self.clusters[0] if len(self.clusters) > 0 else None

    def _validate_all(self):
        self._validate_quantile()
        self._validate_ratio()
        self._validate_fitter_type()

    def _validate_fitter_type(self):
        if self.metric_type == MetricType.ratio and self.fitter_type not in [
            FitterType.bootstrap,
            FitterType.basic,
        ]:
            logger.warning(
                f"""
                causal-platform only support basic and bootstrap fitter type for ratio metric,
                will use basic fitter type in the analysis for ratio metric '{self.column_name}'
                """
            )
            self.fitter_type = FitterType.basic

        if self.metric_aggregate_func == MetricAggregateFunc.quantile and self.fitter_type == FitterType.regression:
            logger.warning(
                f"""
                causal-platform only support basic and bootstrap fitter type for quantile metric,
                will use basic fitter type in the analysis for quantile metric '{self.column_name}'
                """
            )
            self.fitter_type = FitterType.basic

    def _validate_quantile(self):
        if self.metric_aggregate_func == MetricAggregateFunc.quantile and self.quantile is None:
            raise InputConfigError("Must provide quantile argument if metric_aggregate_func is quantile!")
        if self.quantile is not None and (self.quantile < 0 or self.quantile > 1):
            raise InputConfigError("Quantile value must be between 0 and 1!")

    def _validate_ratio(self):
        if self.metric_type == MetricType.ratio:
            if self.numerator_column is None or self.denominator_column is None:
                raise InputConfigError(
                    f"Please provide numerator and denominator columns for ratio metric '{self.column_name}' in config."
                )

            if self.metric_aggregate_func != MetricAggregateFunc.mean:
                raise InputConfigError("We only support aggregate function 'mean' for ratio metric.")

    def __str__(self):
        return f"Metric {self.column_name}"

    def __repr__(self):
        return f"Metric({self._get_object_summary()})"


class DateColumn(Column):
    def __init__(
        self,
        column_name: str,
        date_format: Optional[str] = None,
    ):
        super().__init__(column_name, ColumnType.date)
        self.date_format = date_format


class ExperimentVariation:
    def __init__(self, variation_name: Union[str, int, float], variation_split: float):
        # variation_name is the name of the variation, for example "control", "treatment"
        self.variation_name = variation_name
        # variation_split is the proprotion of this variation in the experiment design
        # i.e 0.5
        self.variation_split = variation_split
        self._validate_variation_split()

    def _validate_variation_split(self):
        if not isinstance(self.variation_split, float):
            raise InputConfigError("variation split must be of type float!")

        if self.variation_split >= 1 or self.variation_split <= 0:
            raise InputConfigError("The value of variation split must be between 0 and 1!")


class ExperimentGroup(Column):
    def __init__(
        self,
        column_name: str,
        control: ExperimentVariation = ExperimentVariation(
            Constants.EXPERIMENTVARIATION_DEFAULT_CONTROL_NAME,
            Constants.EXPERIMENTVARIATION_DEFAULT_CONTROL_SPLIT,
        ),
        treatments: List[ExperimentVariation] = [
            ExperimentVariation(
                Constants.EXPERIMENTVARIATION_DEFAULT_TREATMENT_NAME,
                Constants.EPXERIMENTVARIATION_DEFAULT_TREATMENT_SPLIT,
            )
        ],
    ):
        super().__init__(column_name, ColumnType.experiment_group)
        self.control = control
        self.treatments = treatments
        self._validate_variation_split()

    def __str__(self):
        return f"ExperimentGroupColumn {self.column_name}"

    def __repr__(self):
        return f"ExperimentGroupColumn({self._get_object_summary()})"

    @property
    def all_variation_names(self) -> List[str]:
        # have control in the first place to align with simulation
        variation_names = []
        variation_names.append(self.control.variation_name)
        for treatment_variation in self.treatments:
            variation_names.append(treatment_variation.variation_name)
        return variation_names

    @property
    def all_variation_names_excl_control(self) -> List[str]:
        variation_names = []
        for treatment_variation in self.treatments:
            variation_names.append(treatment_variation.variation_name)
        return variation_names

    @property
    def all_variation_splits(self) -> List[float]:
        # have control in the first place to align with simulation
        variation_splits = []
        variation_splits.append(self.control.variation_split)
        for treatment_variation in self.treatments:
            variation_splits.append(treatment_variation.variation_split)
        return variation_splits

    @property
    def all_variations(self) -> List[ExperimentVariation]:
        # have control in the first place to align with simulation
        return [self.control] + self.treatments

    def _validate_variation_split(self):
        tolerance = 1e-10
        if not (1 - tolerance < np.sum(self.all_variation_splits) < 1 + tolerance):
            raise InputConfigError(
                f"Sum of variation split must equal 1! \
                    Sum of experiment group {self.column_name} is not equal to 1!"
            )


class SimulationMetric:
    def __init__(
        self,
        metric_name: str,
        treatment_effect_mean: float,
        treatment_effect_std: float,
        metric: Metric = None,
    ):
        self.metric_name = metric_name
        self.treatment_effect_mean = treatment_effect_mean
        self.treatment_effect_std = treatment_effect_std
        self.metric = metric
