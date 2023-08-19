"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from causal_platform.src.models.message.message import Message, MessageCollection


class TestResult:
    def test_message_collection_add_messages(self):
        message_collection = MessageCollection()

        message_collection.add_overall_message(Message(title="1"))
        message_collection.add_overall_message(Message(title="2"))
        message_collection.add_overall_message(Message(title="3"))

        message_collection.add_metric_message("col_1", Message(title="4"))
        message_collection.add_metric_message("col_1", Message(title="5"))
        message_collection.add_metric_message("col_2", Message(title="6"))

        assert len(message_collection.overall_messages) == 3
        assert len(message_collection.metric_messages) == 2
        assert len(message_collection.metric_messages["col_1"]) == 2
        assert len(message_collection.metric_messages["col_2"]) == 1

    def test_preprocess_pipeline_add_messages(self):
        message_collection1 = MessageCollection()
        message_collection2 = MessageCollection()

        message_collection1.add_overall_message(Message(title="1"))
        message_collection1.add_overall_message(Message(title="2"))
        message_collection1.add_overall_message(Message(title="3"))
        message_collection2.add_overall_message(Message(title="4"))

        message_collection1.add_metric_message("col_1", Message(title="4"))
        message_collection1.add_metric_message("col_2", Message(title="5"))
        message_collection1.add_metric_message("col_3", Message(title="6"))
        message_collection2.add_metric_message("col_1", Message(title="7"))

        message_collection1.combine(message_collection2)

        assert len(message_collection1.overall_messages) == 4
        assert len(message_collection1.metric_messages) == 3
        assert len(message_collection1.metric_messages["col_1"]) == 2
        assert len(message_collection1.metric_messages["col_2"]) == 1
        assert len(message_collection1.metric_messages["col_3"]) == 1
