"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import copy
from typing import List, Optional, Tuple

import pandas as pd
import statsmodels.formula.api as smf

from causal_platform.src.models.configuration_model.base_objects import (
    Column,
    ColumnType,
    Covariate,
    CovariateType,
    ExperimentGroup,
    Metric,
)
from causal_platform.src.models.message.message import MessageCollection
from causal_platform.src.models.result_model.result import AnalysisResult
from causal_platform.src.utils.error import InputConfigError
from causal_platform.src.utils.experiment.fitter_utils import (
    get_covariates_by_type,
    get_demean_column_name,
    process_data_for_fitter,
)
from causal_platform.src.utils.experiment.result_utils import (
    calculate_basic_sample_metric_stats,
    calculate_interaction_metric_stats,
    get_variation_data,
)
from causal_platform.src.utils.experiment.use_t import adjust_result_using_t
from causal_platform.src.utils.logger import logger


class Fitter:
    def fit(self, *args, **kwargs):
        raise NotImplementedError("No implementation of fit method in class Fitter")

    def get_analysis_results(self) -> List[AnalysisResult]:
        raise NotImplementedError("No implementation of get_analysis_result method in class Fitter")


class SMFFitter(Fitter):
    def __init__(
        self,
        data,
        metric: Metric,
        experiment_groups: List[ExperimentGroup],
        covariates: Optional[List[Covariate]] = None,
        interaction: bool = False,
        cluster: Optional[Column] = None,
        formula: str = None,
        use_t: bool = False,
        use_t_df_adjustment: Optional[int] = None,
        fixed_effect_estimator: bool = False,
    ):
        self.message_collection = MessageCollection()
        self.metric = copy.deepcopy(metric)
        self.experiment_groups = copy.deepcopy(experiment_groups)
        self.covariates = copy.deepcopy(covariates) if covariates is not None else []
        self.cluster = copy.deepcopy(cluster)
        self.interaction = interaction
        if self.interaction and len(self.experiment_groups) > 2:
            raise InputConfigError(
                "causal-platform only support two way interaction! "
                "There are more than two experiment_groups in the config."
            )
        if use_t and use_t_df_adjustment is None:
            raise InputConfigError("If use_t is True, must provide use_t_df_adjustment!")
        self.use_t = use_t
        self.use_t_df_adjustment = use_t_df_adjustment
        self.fixed_effect_estimator = fixed_effect_estimator

        self.data = process_data_for_fitter(
            self.experiment_groups + self.covariates + [self.cluster] + [self.metric], data
        )

        if formula:
            logger.info("SMFFitter: use user specified formula")
            self.formula = formula

        if self.fixed_effect_estimator:
            (
                self.categorical_covariates,
                self.numerical_covariates,
                _,
            ) = get_covariates_by_type(self.covariates)
            self.numerical_covariate_names = [num_cov.column_name for num_cov in self.numerical_covariates]
            self.categorical_covariate_names = [cat_cov.column_name for cat_cov in self.categorical_covariates]

        if self.fixed_effect_estimator and len(self.categorical_covariates) > 0:
            self._demean_fixed_effect()
            self.is_demean_used = True
            self.formula = self._generate_formula_from_columns_with_demean()
            self.use_t = True
            self.use_t_df_adjustment = self.count_formula_terms + self.absorb_columns + 1
        else:
            self.formula = self._generate_formula_from_columns()
            self.is_demean_used = False

        self.method = smf.ols(self.formula, data=self.data)
        self.model = None

    def fit(self, *args, **kwargs):
        if self.cluster is not None:
            self._fit_crse(self.cluster.column_name, *args, **kwargs)
        else:
            self._fit_ols(*args, **kwargs)

    def get_two_way_interaction_results(self) -> List[AnalysisResult]:
        results = []
        exp_group_1st = self.experiment_groups[0]
        exp_group_2nd = self.experiment_groups[1]
        for term1, variation1 in self._get_ols_model_result_key_list(exp_group_1st):
            for term2, variation2 in self._get_ols_model_result_key_list(exp_group_2nd):
                (control_value, control_size, control_data_size,) = calculate_interaction_metric_stats(
                    self.data,
                    self.metric,
                    exp_group_1st,
                    exp_group_2nd,
                    exp_group_1st.control,
                    exp_group_2nd.control,
                    variation2,
                )

                (treatment_value, treatment_size, treatment_data_size,) = calculate_interaction_metric_stats(
                    self.data,
                    self.metric,
                    exp_group_1st,
                    exp_group_2nd,
                    variation1,
                    exp_group_2nd.control,
                    variation2,
                )

                term = "{}:{}".format(term1, term2)
                if term in self.model.params.index:
                    (
                        estimated_treatment_effect,
                        p_value,
                        conf_int,
                        se,
                    ) = self._extract_analysis_result(term)
                    analysis_result = AnalysisResult(
                        metric=self.metric,
                        estimated_treatment_effect=estimated_treatment_effect,
                        p_value=p_value,
                        confidence_interval_left=conf_int[0],
                        confidence_interval_right=conf_int[1],
                        experiment_group=(exp_group_1st, exp_group_2nd),
                        experiment_group_variation=(variation1, variation2),
                        se=se,
                        metric_treatment_value=treatment_value,
                        metric_treatment_sample_size=treatment_size,
                        metric_treatment_data_size=treatment_data_size,
                        metric_control_value=control_value,
                        metric_control_sample_size=control_size,
                        metric_control_data_size=control_data_size,
                        is_interaction_result=True,
                    )
                    if self.use_t:
                        analysis_result = self._degree_of_freedom_adjustment(analysis_result)

                    results.append(analysis_result)

        return results

    def get_analysis_results(self) -> List[AnalysisResult]:
        results = []

        if self.interaction:
            exp_group_1st = self.experiment_groups[0]
            exp_group_2nd = self.experiment_groups[1]
            analysis_result = self._get_analysis_result(
                get_variation_data(self.data, exp_group_2nd, exp_group_2nd.control), exp_group_1st
            )
            results.extend(analysis_result)
            analysis_result = self._get_analysis_result(
                get_variation_data(self.data, exp_group_1st, exp_group_1st.control), exp_group_2nd
            )
            results.extend(analysis_result)
            interaction_results = self.get_two_way_interaction_results()
            results.extend(interaction_results)
        else:
            for grp in self.experiment_groups:
                results.extend(self._get_analysis_result(self.data, grp))

        return results

    def _get_analysis_result(self, data: pd.DataFrame, exp_group: ExperimentGroup) -> List[AnalysisResult]:
        results = []
        ols_key_list = self._get_ols_model_result_key_list(exp_group)
        control_data = data[data[exp_group.column_name] == exp_group.control.variation_name]
        control_value, control_size, control_data_size = calculate_basic_sample_metric_stats(control_data, self.metric)
        for model_result_key, variation in ols_key_list:
            treatment_data = data[data[exp_group.column_name] == variation.variation_name]
            (
                treatment_value,
                treatment_size,
                treatment_data_size,
            ) = calculate_basic_sample_metric_stats(treatment_data, self.metric)

            (
                estimated_treatment_effect,
                p_value,
                conf_int,
                se,
            ) = self._extract_analysis_result(model_result_key)
            analysis_result = AnalysisResult(
                metric=self.metric,
                estimated_treatment_effect=estimated_treatment_effect,
                p_value=p_value,
                confidence_interval_left=conf_int[0],
                confidence_interval_right=conf_int[1],
                experiment_group=exp_group,
                experiment_group_variation=variation,
                se=se,
                metric_treatment_value=treatment_value,
                metric_treatment_sample_size=treatment_size,
                metric_treatment_data_size=treatment_data_size,
                metric_control_value=control_value,
                metric_control_sample_size=control_size,
                metric_control_data_size=control_data_size,
            )
            if self.use_t:
                analysis_result = self._degree_of_freedom_adjustment(analysis_result)

            results.append(analysis_result)

        return results

    def _degree_of_freedom_adjustment(self, analysis_result: AnalysisResult):
        """
        statsmodels uses the normal distribution by default for robust standard error.
        When we have small sample size and low degrees of freedom, the critical
        values of the t-distribution differ from those of the normal distribution
        in an observable magnitude. And this can cause high false positive in DID
        analysis. The following replaces the critical value of 1.96 by the critical
        values from the t-distribution with the degree of freedom based on DID formula
        """
        if self.cluster is None:
            n = self.data.shape[0]
        else:
            n = self.data[self.cluster.column_name].nunique()

        p_value, ci_left, ci_right = adjust_result_using_t(
            est=analysis_result.estimated_treatment_effect,
            se=analysis_result.se,
            n=n,
            p=self.use_t_df_adjustment,
            alpha=analysis_result.metric.alpha,
        )
        # override p-values, confidence interval
        analysis_result.p_value = p_value
        analysis_result.confidence_interval_left = ci_left
        analysis_result.confidence_interval_right = ci_right
        return analysis_result

    def _extract_analysis_result(self, model_result_key: str):
        estimated_treatment_effect = self.model.params[model_result_key]
        p_value = self.model.pvalues[model_result_key]
        conf_int = self.model.conf_int(alpha=self.metric.alpha).loc[model_result_key]
        se = self.model.bse[model_result_key]

        return (
            estimated_treatment_effect,
            p_value,
            conf_int,
            se,
        )

    def _fit_crse(self, cluster_column_name, *args, **kwargs):
        self.model = self.method.fit(
            cov_type="cluster",
            cov_kwds={"groups": self.data[cluster_column_name]},
            *args,
            **kwargs,
        )

    def _fit_ols(self, *args, **kwargs):
        self.model = self.method.fit(*args, **kwargs)

    def _generate_formula_from_columns(self):
        terms_to_add = []
        added_columns = set()
        if self.interaction:
            columns_tuple = (self.experiment_groups[0], self.experiment_groups[1])
            terms_to_add.append(self._get_formula_element_from_interaction(columns_tuple))
            for col in columns_tuple:
                added_columns.add(col)

        for grp in self.experiment_groups:
            if grp not in added_columns:
                terms_to_add.append(self._get_formula_element_from_column(grp))

        if self.covariates:
            for cov in self.covariates:
                if cov not in added_columns:
                    terms_to_add.append(self._get_formula_element_from_column(cov))

        return "{} ~ {}".format(self.metric.column_name, " + ".join(terms_to_add))

    def _generate_formula_from_columns_with_demean(self):

        terms_to_add = [
            get_demean_column_name(term)
            for term in self.experiment_groups_dummies_name + self.numerical_covariate_names
        ]
        demean_metric_name = get_demean_column_name(self.metric.column_name)

        self.count_formula_terms = len(terms_to_add) + 1
        return "{} ~ {}".format(demean_metric_name, " + ".join(terms_to_add))

    def _get_formula_element_from_column(self, column: Column) -> str:
        """
        get the entry in standard R formula
        :param column:
        :return:
        """
        # TODO(Sifeng) consider move this into Column function if it is standard across packages
        if column.column_type == ColumnType.experiment_group:
            control_label = column.control.variation_name
            if type(control_label) == str:
                control_label_str = "'{}'".format(control_label)
            else:
                control_label_str = "{}".format(control_label)
            return "C({}, Treatment({}))".format(column.column_name, control_label_str)
        elif column.column_type == ColumnType.covariate and column.value_type == CovariateType.categorial:
            return "C({})".format(column.column_name)
        else:
            return column.column_name

    def _get_formula_element_from_interaction(self, columns: Tuple[Column]) -> str:
        return "*".join(self._get_formula_element_from_column(col) for col in columns)

    def _get_ols_model_result_key_list(self, column: Column) -> List[Tuple]:
        """
        get the formula string element for a column value:
            for numerical covariate, it will be the column name
            for categorical variable and experiment group, add the column value to column formula
        :param column:
        :return: a list of tuples, each tuple has two elements,
            first element is the formula element value,
            second one is the corresponding value in the column
        """

        def get_experiment_group_ols_key(exp_group: ExperimentGroup, is_demean_used: bool = False):
            ols_key_list = []
            for variation in exp_group.treatments:
                if is_demean_used:
                    column_name = "{}_{}".format(column.column_name, variation.variation_name)
                    ols_key = get_demean_column_name(column_name)
                else:
                    formula_str = self._get_formula_element_from_column(column)
                    ols_key = "{}[T.{}]".format(formula_str, variation.variation_name)
                ols_key_list.append((ols_key, variation))
            return ols_key_list

        def get_categorical_covariate_ols_key(column: Column, is_demean_used: bool = False):
            if is_demean_used:
                raise InputConfigError("Fixed effect {} has been absorbed!".format(column.column_name))
            else:
                all_covariate_variations = self.data[column.column_name].unique()
                ols_key_list = [("{}[T.{}]".format(column.column_name, val), val) for val in all_covariate_variations]
            return ols_key_list

        def get_other_ols_key(column: Column, is_demean_used: bool = False):
            if is_demean_used:
                ols_key_list = [(get_demean_column_name(column.column_name), None)]
            else:
                ols_key_list = [(column.column_name, None)]
            return ols_key_list

        is_experiment_group = column.column_type == ColumnType.experiment_group
        is_categorical_covariate = (
            column.column_type == ColumnType.covariate and column.value_type == CovariateType.categorial
        )

        if is_experiment_group:
            ols_key_list = get_experiment_group_ols_key(column, self.is_demean_used)
        elif is_categorical_covariate:
            ols_key_list = get_categorical_covariate_ols_key(column, self.fixed_effect_estimator)
        else:
            ols_key_list = get_other_ols_key(column, self.is_demean_used)

        return ols_key_list

    def _demean_fixed_effect(self):
        if self.interaction:
            raise InputConfigError(
                """
                When fixed effect demean method is used, interaction effect are not able to calculate!
                """
            )

        if len(self.categorical_covariates) > 1:
            raise InputConfigError(
                """
                causal-platform only support fixed effect estimator when there is no more than one categorical variable.
                """
            )

        # step1: convert experiment groups to dummies
        self._dummy_encode_experiment_groups()

        # step2: create new columns in data and demean
        original_columns_to_demean = (
            [self.metric.column_name] + self.experiment_groups_dummies_name + self.numerical_covariate_names
        )

        demean_column_names = [get_demean_column_name(col) for col in original_columns_to_demean]
        # TODO: use original columns with a postfix as the new columns names.
        # Might run into error if there are duplicate column name.
        self.data[demean_column_names] = self.data[original_columns_to_demean]
        self.absorb_columns = 0
        for fe in self.categorical_covariate_names:
            self.data[demean_column_names] = self.data[demean_column_names] - self.data.groupby(fe)[
                demean_column_names
            ].transform("mean")
            num_levels = self.data[fe].nunique()
            self.absorb_columns += num_levels - 1

    def _dummy_encode_experiment_groups(self):
        experiment_groups_dummies_name = []
        for group in self.experiment_groups:
            dummies = pd.get_dummies(self.data[group.column_name], drop_first=False, prefix=group.column_name)
            control_column = "{}_{}".format(group.column_name, group.control.variation_name)
            dummies.drop(control_column, axis=1, inplace=True)
            experiment_groups_dummies_name.extend(dummies.columns)
            self.data = pd.concat([self.data, dummies], axis=1)
        self.experiment_groups_dummies_name = experiment_groups_dummies_name
