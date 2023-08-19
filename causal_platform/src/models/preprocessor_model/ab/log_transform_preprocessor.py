"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
import pandas as pd

from causal_platform.src.models.data_model import DataLoader
from causal_platform.src.models.message.message import Message, Source, Status
from causal_platform.src.models.preprocessor_model.base import BasePreprocessor
from causal_platform.src.models.result_model.result import PreprocessResult


class LogTransformPreprocessor(BasePreprocessor):
    def __init__(self, column_name: str):
        self.column_name = column_name

    def process(self, data: pd.DataFrame) -> PreprocessResult:
        preprocess_result = PreprocessResult(data)

        # check if data < 0
        if (data[self.column_name] <= 0).any():
            # TODO(caixia): should we raise error or just warning and skip log transformation

            description = f"""{self.column_name} column contains zero or negative value and can't log transformation!
Log transformation is skipped!"""

            preprocess_result.message_collection.add_metric_message(
                self.column_name,
                Message(
                    source=Source.transformation,
                    title="Log transformation",
                    description=description,
                    status=Status.skip,
                ),
            )
        else:
            preprocess_result.message_collection.add_metric_message(
                self.column_name,
                Message(
                    source=Source.transformation,
                    title="Log transformation",
                    description=f"Log transformation applied to {self.column_name}",
                    status=Status.success,
                ),
            )

            data[self.column_name] = np.log(data[self.column_name])
            preprocess_result.processed_data = data
        return preprocess_result


class LowMemLogXformPreprocessor(BasePreprocessor):
    def __init__(self, column_name: str):
        self.column_name = column_name

    def process(self, data: DataLoader) -> PreprocessResult:
        preprocess_result = PreprocessResult(data)

        # read data once here to check if any values are lte zero and again later when we overwrite
        col_subset = data[self.column_name]

        if col_subset.min() <= 0:
            description = f"{self.column_name} column contains 0 or negative, so it is skipped!"
            msg = Message(Source.transformation, "Log transformation", description, Status.skip)
            preprocess_result.message_collection.add_metric_message(self.column_name, msg)

        else:
            description = f"Log transformation applied to {self.column_name}"
            msg = Message(Source.transformation, "Log transformation", description, Status.success)
            preprocess_result.message_collection.add_metric_message(self.column_name, msg)

            data.set_columns({self.column_name: np.log})

        return preprocess_result
