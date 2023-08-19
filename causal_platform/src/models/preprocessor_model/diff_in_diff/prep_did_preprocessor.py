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

from causal_platform.src.models.configuration_model.config import DiDConfig
from causal_platform.src.models.preprocessor_model.base import BasePreprocessor
from causal_platform.src.models.result_model.result import PreprocessResult
from causal_platform.src.utils.diff_in_diff.prep_data import (
    get_data_between_start_end_date,
    prep_data_for_diff_in_diff,
)


class PrepDiDPreprocessor(BasePreprocessor):
    def __init__(self, config: DiDConfig, control_unit_ids: List[int]):
        self.config = config
        self.control_unit_ids = control_unit_ids

    def process(self, data: pd.DataFrame) -> PreprocessResult:
        """prepare data for diff in diff analysis"""
        experiment_data = get_data_between_start_end_date(
            data,
            self.config.date.column_name,
            self.config.matching_start_date,
            self.config.experiment_end_date,
        )

        diff_in_diff_data = prep_data_for_diff_in_diff(
            experiment_data,
            self.config.treatment_unit_ids,
            self.control_unit_ids,
            self.config.experiment_randomize_units[0].column_name,
            self.config.date.column_name,
            self.config.experiment_start_date,
        )

        preprocess_result = PreprocessResult(processed_data=diff_in_diff_data)
        return preprocess_result
