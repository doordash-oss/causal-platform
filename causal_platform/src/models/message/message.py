"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Dict, List, Optional

from causal_platform.src.models.message.enums import Source, Status
from causal_platform.src.utils.logger import logger


class Message:
    def __init__(
        self,
        source: Optional[Source] = None,
        status: Optional[Status] = None,
        title: str = "",
        description: str = "",
    ):
        """
        source: str
            Where is the message originating from?
                pre-processing
                analysis

        status: str
            What is the status?
                pass
                fail

        title: str
            Short description

        description: str
            More detailed description

        """

        self.source = source
        self.status = status
        self.title = title
        self.description = description

    def __str__(self):
        return f"""
        Message:
            Source:      {self.source}
            Status:      {self.status}
            Title:       {self.title}
            Description: {self.description}
        """


class MessageCollection:
    def __init__(self):
        self.overall_messages: List[Message] = []
        self.metric_messages: Dict[str, List[Message]] = {}

    def add_overall_message(self, message: Message):
        self.log_message(message)
        self.overall_messages.append(message)

    def add_metric_message(self, metric: str, message: Message):
        self.log_message(message)

        if metric not in self.metric_messages:
            self.metric_messages[metric] = []

        self.metric_messages[metric].append(message)

    def combine(self, message_collection: "MessageCollection"):
        self.overall_messages.extend(message_collection.overall_messages)

        for metric, messages in message_collection.metric_messages.items():
            if metric not in self.metric_messages:
                self.metric_messages[metric] = []

            self.metric_messages[metric].extend(messages)

    def log_message(self, message: Message):
        if message.status == Status.warn or message.status == Status.skip:
            logger.warning(message.description)
        elif message.status == Status.fail:
            logger.error(message.description)
        else:
            logger.info(message.description)

    def get_all_messages(self) -> List[Message]:
        overall_messages_list = self.overall_messages
        metric_messages = list(self.metric_messages.values())
        metric_messages_list = [item for sublist in metric_messages for item in sublist]
        return overall_messages_list + metric_messages_list

    def get_messages_description_list(self) -> Dict[Status, List[str]]:
        all_messages = self.get_all_messages()
        return {
            Status.warn: [
                message.description
                for message in all_messages
                if message.status == Status.warn and message.description != ""
            ],
            Status.fail: [
                message.description
                for message in all_messages
                if message.status == Status.fail and message.description != ""
            ],
        }
