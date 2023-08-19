"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from causal_platform.src.models.configuration_model.base_objects import SuperEnum


class Source(SuperEnum):
    validation = "validation"
    transformation = "transformation"
    analysis = "analysis"


class Status(SuperEnum):
    success = "success"
    fail = "fail"
    warn = "warn"
    skip = "skip"
