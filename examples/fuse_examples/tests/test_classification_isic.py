"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

import unittest
import os
import tempfile
import shutil
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess

from fuse_examples.imaging.classification.isic.runner import (
    TRAIN_COMMON_PARAMS,
    INFER_COMMON_PARAMS,
    EVAL_COMMON_PARAMS,
    run_train,
    run_infer,
    run_eval,
    PATHS,
)
from fuse_examples.imaging.classification.isic.golden_members import FULL_GOLDEN_MEMBERS

import fuse.utils.gpu as GPU
from fuse.utils.rand.seed import Seed
from fuseimg.datasets.isic import ISIC


class ClassificationISICTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

        self.paths = {
            "model_dir": os.path.join(self.root, "isic/model_dir"),
            "force_reset_model_dir": True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            "data_dir": PATHS["data_dir"],
            "cache_dir": os.path.join(self.root, "isic/cache_dir"),
            "inference_dir": os.path.join(self.root, "isic/infer_dir"),
            "eval_dir": os.path.join(self.root, "isic/eval_dir"),
        }

        self.train_common_params = TRAIN_COMMON_PARAMS
        self.train_common_params["manager.train_params"]["num_epochs"] = 15
        self.train_common_params["samples_ids"] = FULL_GOLDEN_MEMBERS

        self.infer_common_params = INFER_COMMON_PARAMS
        self.eval_common_params = EVAL_COMMON_PARAMS

        ISIC.download(data_path=self.paths["data_dir"])

    @run_in_subprocess
    def test_runner(self):
        # Must use GPU due a long running time
        GPU.choose_and_enable_multiple_gpus(1, use_cpu_if_fail=False)

        Seed.set_seed(
            0, False
        )  # previous test (in the pipeline) changed the deterministic behavior to True

        run_train(self.paths, self.train_common_params)
        run_infer(self.paths, self.infer_common_params)
        results = run_eval(self.paths, self.eval_common_params)

        threshold = 0.65
        self.assertGreaterEqual(results["metrics.auc"], threshold)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
