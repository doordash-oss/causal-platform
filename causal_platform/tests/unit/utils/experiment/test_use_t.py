"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from causal_platform.src.utils.experiment.use_t import adjust_result_using_t


class TestUseT:
    def test_adjust_result_using_t(self):

        est = 3
        se = 1
        num_data = 100
        num_predictor = 5
        p_value, ci_left, ci_right = adjust_result_using_t(est, se, num_data, num_predictor, alpha=0.05)
        assert ci_right > ci_left
        assert ci_left > 0
        assert p_value <= 0.05
