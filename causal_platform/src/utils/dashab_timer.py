"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

def time_profiler(process_name):
    def inner_time_function(func):
        def wrapper(*args, **kwargs):
            timer_callback = kwargs.get("timer_callback")
            if timer_callback:
                with timer_callback(process_name):
                    response = func(*args, **kwargs)
            else:
                response = func(*args, **kwargs)
            return response

        return wrapper

    return inner_time_function
