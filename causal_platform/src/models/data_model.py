"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow.parquet as pq


class DataLoader:
    """Simple class to make parquet folders look like dataframes and vice versa."""

    def __init__(self, data_folder: Path):
        # easy to forget that path should not be string, so assert here
        assert isinstance(data_folder, Path)

        self.data_folder = data_folder

    def set_columns(self, column_ops: Dict):
        """
        Overwrite columns using specified operation in col-op pair, where op is operation to overwrite column.
        Columns must be applying a 1-1 transformation overwriting themselves for log xform.
        """

        for file in self.data_folder.iterdir():
            data = pd.read_parquet(file)
            for col, func in column_ops.items():
                data[col] = func(data[col])
            data.to_parquet(file)

    def copy(self):
        return DataLoader(data_folder=self.data_folder)

    def force_cols_to_number_or_date(
        self,
        num_columns: List[str],
        date_columns: List[str],
        date_formats: Optional[List[str]] = None,
    ):

        # first check if this is necessary (bc it is expensive and can be done in advance).
        curr_type_map = {c["name"]: c["pandas_type"] for c in self.parquet_metadata["columns"]}

        # check if any single column needs to be retyped
        retype_parquet_files = False
        for col, typ in curr_type_map.items():
            if col in num_columns and typ[:3] not in ["flo", "int"]:  # float or int
                retype_parquet_files = True
            if col in date_columns and typ[:4] not in ["date"]:  # datetime or date (?)
                retype_parquet_files = True

        # if we decide to retype, we have to iterate over entire parquet directory
        if retype_parquet_files:

            # check for problems with mismatched column lists
            if date_formats is not None:
                assert len(date_columns) == len(date_formats)

            for file in self.data_folder.iterdir():
                data = pd.read_parquet(file)
                for col in num_columns:
                    data[col] = pd.to_numeric(data[col])
                for i, col in enumerate(date_columns):
                    if date_formats is None:
                        data[col] = pd.to_datetime(data[col])
                    else:
                        data[col] = pd.to_datetime(data[col], date_formats[i])

                data.to_parquet(file)

    # validation functions are very long and intricately tied to pandas
    # so we want to spoof relevant syntax to minimize changes necessary
    def __getitem__(self, columns: Union[str, List[str]]):
        """Return columns of interest in pandas format."""
        if isinstance(columns, str):
            return pd.read_parquet(self.data_folder, columns=[columns]).squeeze()
        else:
            deduped_columns = list(set(columns))
            return pd.read_parquet(self.data_folder, columns=deduped_columns)

    def __len__(self):
        return self.shape[0]

    @property
    def columns(self):
        # return in same format as we'd see if we used pandas
        return pd.Index([c["name"] for c in self.parquet_metadata["columns"]])

    @columns.setter
    def columns(self, new_columns):
        # check if any single column needs to be renamed
        if not all(self.columns == new_columns):
            for file in self.data_folder.iterdir():
                data = pd.read_parquet(file)
                data.columns = new_columns
                data.to_parquet(file)

    @property
    def shape(self):
        height = 0
        for file in self.data_folder.iterdir():
            height += pq.read_metadata(file).num_rows
        width = pq.read_metadata(file).num_columns

        return height, width

    @property
    def dtypes(self):
        metadata_cols = self.parquet_metadata["columns"]
        metadata_dtypes = {c["name"]: c["numpy_type"] for c in metadata_cols}
        return pd.Series(metadata_dtypes)

    @property
    def parquet_metadata(self):
        file = next(self.data_folder.iterdir())
        metadata = pq.read_metadata(file)
        metadata_cleartext = metadata.schema.to_arrow_schema().metadata[b"pandas"]
        return json.loads(metadata_cleartext)
