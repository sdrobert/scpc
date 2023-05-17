#!/usr/bin/env bash

# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# As this project has developed, new hyperparameters have been added to
# model.yaml files. This script goes through all model.yaml files and joins the
# parameters with a list of defaults. This should introduce any new
# hyperparameters with default values.
#
# WARNING: If a new hyperparameter default does not match what was being done
# when the model was initially specified, i.e. the hyperparameter was a
# different value implicitly, then this logic will be incorrect.

exp="${1:-exp}"

if ! which scpc-train > /dev/null ; then
  echo "scpc-train command cannot be found (is the environment active?)"
  exit 1
fi

set -e

for fn in $(find "$exp/" -name 'model.yaml'); do
  echo "$fn"
  scpc-train --print-model-yaml | \
    combine-yaml-files --quiet --nested \
      - "$fn"{,_}
  mv "$fn"{_,}
done
