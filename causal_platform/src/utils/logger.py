"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import datetime
import logging
import sys
from typing import Optional


class DashABLogger(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DashABLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._set_stream_handler()
        self.logger.propagate = False

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def _set_stream_handler(self):
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.handlers = [handler]

    def _set_file_handler(self, prefix):
        log_timestamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        handler = logging.FileHandler(prefix + "_" + log_timestamp)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.handlers = [handler]

    def reset_logger_with_type(self, logger_type: Optional[str], prefix: str = ""):
        """
        reset with logger type to file or stream, if nothing passed,
        keep the current logger type
        """
        if logger_type == "file":
            self._set_file_handler(prefix)
        elif logger_type == "stream":
            self._set_stream_handler()


logger = DashABLogger()
