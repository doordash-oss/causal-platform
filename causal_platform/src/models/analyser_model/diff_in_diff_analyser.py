"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import List

from causal_platform.src.models.analyser_model.analyser import BaseAnalyser
from causal_platform.src.models.analyser_model.fitters.fitter import SMFFitter
from causal_platform.src.models.configuration_model.base_objects import (
    ExperimentGroup,
    ExperimentVariation,
)
from causal_platform.src.models.configuration_model.config import DiDConfig
from causal_platform.src.models.result_model.result import (
    AnalysisResult,
    DiffinDiffPreprocessPipelineResult,
)
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.logger import logger


class DiffinDiffAnalyser(BaseAnalyser):
    def __init__(self, config: DiDConfig, preprocess_result: DiffinDiffPreprocessPipelineResult):
        super().__init__(config, preprocess_result)

    def _analyze_metric(self, metric) -> List[AnalysisResult]:
        # TODO(caixia): check whether preprocess_result contain non-empty matched_markets

        treatment_term = ExperimentGroup(
            Constants.DIFF_IN_DIFF_TREATMENT,
            control=ExperimentVariation(variation_name=0, variation_split=0.5),
            treatments=[ExperimentVariation(variation_name=1, variation_split=0.5)],
        )
        time_term = ExperimentGroup(
            Constants.DIFF_IN_DIFF_TIME,
            control=ExperimentVariation(variation_name=0, variation_split=0.5),
            treatments=[ExperimentVariation(variation_name=1, variation_split=0.5)],
        )

        if self.config.small_sample_adjustment and self._is_small_sample:
            logger.info("Small sample enabled and detected, applying t-distribution to correct.")
            use_t = True
        else:
            use_t = False

        # Use 3 given the that the within_ci is most close to 0.95 from simulation
        degree_of_freedom_adjustment = 3 + len(metric.covariates)

        # get result of y = treatment_term + time_term + treatment_term * time_term
        fitter = SMFFitter(
            data=self.preprocess_result.processed_data,
            metric=metric,
            experiment_groups=[treatment_term, time_term],
            covariates=metric.covariates,
            interaction=True,
            cluster=metric.cluster,
            use_t=use_t,
            use_t_df_adjustment=degree_of_freedom_adjustment,
        )
        fitter.fit()
        results = fitter.get_two_way_interaction_results()
        # TODO: only one element in this return list in diff-in-diff case, but we
        # need to write a function so that user can get the result they want from the list
        return results

    @property
    def _is_small_sample(self):
        # TODO(yixin): add data imbalance check and  more customized
        #  small sample check after simmulation, for more results here:
        #  http://www.ucd.ie/geary/static/publications/workingpapers/gearywp201802.pdf
        return (len(self.config.treatment_unit_ids) + self.config.match_unit_size) <= 7
