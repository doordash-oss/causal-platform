"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import pathlib
import tempfile

import numpy as np

from causal_platform.src.utils.common_utils import (
    DataLoader,
    convert_table_column_to_lower_case,
    format_number,
)
from causal_platform.tests.data import data_generator


class TestCommonUtils:
    def test_format_number(self):
        num = 2.71828182845904523536
        assert format_number(num, 2, format="float") == "2.72"
        assert format_number(num, 2, format="exp") == "2.72e+00"
        assert format_number(num, 2, format="percent") == "271.83%"

    def test_data_loader(self):
        data = data_generator.get_ab_test_data()
        data["pred_asap"] = data["pred_asap"].map(str)
        data.columns = data.columns.str.upper()

        with tempfile.TemporaryDirectory() as temp_dir:
            folder = pathlib.Path(temp_dir)
            data.iloc[: data.shape[0] // 2, :].to_parquet(folder / "raw_data.parquet")
            data.iloc[data.shape[0] // 2 :, :].to_parquet(folder / "raw_data2.parquet")
            data_loader = DataLoader(data_folder=folder)
            assert all(data_loader.columns == data.columns)

            data.columns = data.columns.str.lower()
            convert_table_column_to_lower_case(data_loader)
            assert all(data_loader.columns == data.columns)
            assert data.shape == data_loader.shape
            assert len(data) == len(data_loader)
            assert all(data.dtypes == data_loader.dtypes)

            data_loader.force_cols_to_number_or_date(num_columns=["pred_asap"], date_columns=[])
            data["pred_asap"] = data["pred_asap"].astype("float")
            assert np.allclose(data["pred_asap"].values, data_loader["pred_asap"].values)
