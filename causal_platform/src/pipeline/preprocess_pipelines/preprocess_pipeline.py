"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Union

import pandas as pd

from causal_platform.src.models.configuration_model.config import BaseConfig
from causal_platform.src.models.data_model import DataLoader
from causal_platform.src.models.message.message import MessageCollection
from causal_platform.src.models.preprocessor_model.ab.log_transform_preprocessor import (
    LogTransformPreprocessor,
    LowMemLogXformPreprocessor,
)
from causal_platform.src.models.preprocessor_model.base import TBasePreprocessor
from causal_platform.src.models.result_model.result import BasePreprocessPipelineResult, PreprocessResult


class BasePreprocessPipeline:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.message_collection: MessageCollection = MessageCollection()

    def execute_common_preprocesses(self, data: Union[pd.DataFrame, DataLoader]) -> pd.DataFrame:
        result = PreprocessResult(data)
        # Log Transforms
        for metric in self.config.metrics:
            if metric.log_transform:
                if isinstance(data, DataLoader):
                    result = self.execute_preprocess(data, LowMemLogXformPreprocessor(column_name=metric.column_name))
                else:
                    result = self.execute_preprocess(data, LogTransformPreprocessor(column_name=metric.column_name))

        return result.processed_data

    def execute_preprocess(self, data: pd.DataFrame, preprocessor: TBasePreprocessor) -> PreprocessResult:
        result = preprocessor.process(data)
        self.message_collection.combine(result.message_collection)
        return result

    def run(self, data: Union[pd.DataFrame, DataLoader]) -> BasePreprocessPipelineResult:
        raise NotImplementedError()
