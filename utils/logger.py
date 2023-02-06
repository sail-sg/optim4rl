# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging


class Logger(object):
  def __init__(self, logs_dir, file_name='log.txt', filemode='w'):
    logging.basicConfig(
      force=True,
      format='%(asctime)s - %(levelname)s: %(message)s',
      filename=f'{logs_dir}{file_name}',
      filemode=filemode
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    self.debug = logger.debug
    self.info = logger.info
    self.warning = logger.warning
    self.error = logger.error
    self.critical = logger.critical
    self.logs_dir = logs_dir