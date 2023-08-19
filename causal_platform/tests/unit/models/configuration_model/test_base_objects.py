"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import pytest

from causal_platform.src.models.configuration_model.base_objects import (
    ExperimentGroup,
    ExperimentVariation,
)
from causal_platform.src.utils.error import InputConfigError


class TestBaseObjects:
    def test_experiment_variation(self):
        # regular input
        variation = ExperimentVariation(variation_name="group", variation_split=0.5)
        assert variation.variation_name == "group"
        assert variation.variation_split == 0.5
        # wrong data type in config
        with pytest.raises(InputConfigError) as err:
            variation = ExperimentVariation(variation_name=0, variation_split="group")
            assert "variation split must be of type float!" in str(err.value)
        # split not between 0 and 1
        with pytest.raises(InputConfigError) as err:
            variation = ExperimentVariation(variation_name="control", variation_split=1.2)
            assert "between 0 and 1" in str(err.value)

    def test_experiment_group(self):
        # regular input
        group = ExperimentGroup(
            column_name="group",
            control=ExperimentVariation("control", 0.5),
            treatments=[ExperimentVariation("treatment", 0.5)],
        )

        assert group.column_name == "group"
        assert group.control.variation_name == "control"
        assert group.control.variation_split == 0.5
        assert group.treatments[0].variation_name == "treatment"
        assert group.treatments[0].variation_split == 0.5
        assert group.all_variation_names == ["control", "treatment"]
        assert group.all_variation_splits == [0.5, 0.5]
        assert len(group.all_variations) == 2
        assert group.all_variations[0].variation_name == "control"

        # muti-treatment input
        group = ExperimentGroup(
            column_name="group",
            control=ExperimentVariation("control", 0.4),
            treatments=[
                ExperimentVariation("treatment", 0.3),
                ExperimentVariation("treatment2", 0.3),
            ],
        )
        assert group.column_name == "group"
        assert group.control.variation_name == "control"
        assert group.control.variation_split == 0.4
        assert group.treatments[0].variation_name == "treatment"
        assert group.treatments[0].variation_split == 0.3
        assert group.treatments[1].variation_name == "treatment2"
        assert group.treatments[1].variation_split == 0.3
        assert group.all_variation_names == ["control", "treatment", "treatment2"]
        assert group.all_variation_splits == [0.4, 0.3, 0.3]
        assert len(group.all_variations) == 3
        assert group.all_variations[0].variation_name == "control"

        # wrong input
        with pytest.raises(InputConfigError) as err:
            group = ExperimentGroup(
                column_name="group",
                control=ExperimentVariation("control", 0.4),
                treatments=[
                    ExperimentVariation("treatment", 0.3),
                    ExperimentVariation("treatment2", 0.5),
                ],
            )
            assert "must equal 1" in str(err.value)

        # test variation split sum
        group = ExperimentGroup(
            column_name="group",
            control=ExperimentVariation("control", 0.1),
            treatments=[
                ExperimentVariation("treatment1", 0.1),
                ExperimentVariation("treatment2", 0.1),
                ExperimentVariation("treatment3", 0.1),
                ExperimentVariation("treatment4", 0.1),
                ExperimentVariation("treatment5", 0.1),
                ExperimentVariation("treatment6", 0.1),
                ExperimentVariation("treatment7", 0.1),
                ExperimentVariation("treatment8", 0.1),
                ExperimentVariation("treatment9", 0.1),
            ],
        )
