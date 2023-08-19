"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from string import Template
from typing import List, Optional

import pandas as pd
import scipy
from scipy.stats import binom_test, chisquare

from causal_platform.src.models.configuration_model.base_objects import (
    CheckImbalanceMethod,
    Cluster,
    Column,
    ExperimentGroup,
)
from causal_platform.src.models.message.message import Message, Source, Status
from causal_platform.src.models.preprocessor_model.base import BasePreprocessor
from causal_platform.src.models.result_model.result import PreprocessResult
from causal_platform.src.utils.common_utils import format_number
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.logger import logger

MESSAGE_TITLE = "Imbalance Check"


class ImbalancePreprocessor(BasePreprocessor):
    def __init__(
        self,
        experiment_groups: List[ExperimentGroup],
        experiment_randomize_units: Optional[List[Column]],
        cluster: Cluster,
        check_imbalance_method: CheckImbalanceMethod,
    ):
        self.experiment_groups = experiment_groups
        self.experiment_randomize_units = experiment_randomize_units
        self.cluster = cluster
        self.check_imbalance_method = check_imbalance_method
        self.description_template = Template(
            "Bucketing splits in $group_name are deviated from the expected variation splits $splits! "
            "There are in total $size units. The expected size is $expected, while the observed size is $observed."
            "The p-value of the imbalance test is $p_value."
        )

    def process(self, data: pd.DataFrame) -> PreprocessResult:
        preprocess_result = PreprocessResult(data)
        if self.experiment_randomize_units:
            unit = self.experiment_randomize_units[0].column_name
            logger.info(
                f"Running imbalance check on experiment_randomize_unit {self.experiment_randomize_units[0].column_name}"
            )
        elif self.cluster:
            unit = self.cluster.column_name
            logger.info(f"Running imbalance check on cluster {self.cluster.column_name}")
        else:
            unit = None
            logger.info("Running imbalance check on count of data row")

        for group in self.experiment_groups:
            group_name = group.column_name
            logger.info(f"Checking bucket balance in {group_name}...")
            variation_names = [variation.variation_name for variation in group.all_variations]
            splits = [variation.variation_split for variation in group.all_variations]
            if unit is None:
                observed = list(data.groupby(group_name).size().reindex(variation_names))
            else:
                observed = list(data.groupby(group_name)[unit].nunique().reindex(variation_names))
            size = sum(observed)
            expected = [split * size for split in splits]

            if self.check_imbalance_method == CheckImbalanceMethod.chi_square:
                chi2_stats = chisquare(observed, expected)
                if chi2_stats[1] < Constants.IMBALANCE_CHI_SQUARE_THRESHOLD:

                    preprocess_result.are_buckets_imbalanced = True

                    mape_threshold = self.mape_threshold_from_chi_sqaure_test(
                        size, Constants.IMBALANCE_CHI_SQUARE_THRESHOLD, len(group.all_variations)
                    )

                    description = self.description_template.substitute(
                        group_name=group_name,
                        splits=splits,
                        size=size,
                        expected=expected,
                        observed=observed,
                        p_value=format_number(chi2_stats[1], 2, "exp"),
                    )

                    description += " The approximate Mean absolute percentage error(MAPE) threshold is {}.".format(
                        format_number(mape_threshold, 2, "percent")
                    )

                    preprocess_result.message_collection.add_overall_message(
                        Message(
                            source=Source.validation,
                            title=MESSAGE_TITLE,
                            description=description,
                            status=Status.warn,
                        )
                    )

                # needed this check to handle multiple experiment groups
                elif not preprocess_result.are_buckets_imbalanced:
                    preprocess_result.are_buckets_imbalanced = False

            if self.check_imbalance_method == CheckImbalanceMethod.binomial:
                for i in range(len(variation_names)):
                    binom_stats = binom_test(observed[i], size, splits[i])
                    if binom_stats < Constants.IMBALANCE_BINOMIAL_THRESHOLD:
                        preprocess_result.are_buckets_imbalanced = True

                        description = self.description_template.substitute(
                            group_name=group_name,
                            splits=splits,
                            size=size,
                            expected=expected,
                            observed=observed,
                            p_value=format_number(binom_stats, 2, "percent"),
                        )
                        preprocess_result.message_collection.add_overall_message(
                            Message(
                                source=Source.validation,
                                title=MESSAGE_TITLE,
                                description=description,
                                status=Status.warn,
                            )
                        )

                    elif not preprocess_result.are_buckets_imbalanced:
                        preprocess_result.are_buckets_imbalanced = False

            if not preprocess_result.are_buckets_imbalanced:
                preprocess_result.message_collection.add_overall_message(
                    Message(
                        source=Source.validation,
                        title=MESSAGE_TITLE,
                        description=f"Imbalance checked on {size} units",
                        status=Status.success,
                    )
                )

        return preprocess_result

    def mape_threshold_from_chi_sqaure_test(self, n, p_value, cat=2):
        """
        Approximated mean absolute percentage error (MAPE) tolerance using chi-square test
        In the calculation, we assume constant MAPE,
        i.e. approximate sum of (Oi - Ei)^2/Ei by 1/N * MAPE^2 for each i
        :param n: total sample size
        :param p_value: p-value threshold
        :param cat: number of category (group)
        :return: MAPE
        """
        chi_stats = scipy.stats.chi2.ppf(1 - p_value, df=cat - 1)
        percent_error = (chi_stats / n) ** 0.5
        return percent_error
