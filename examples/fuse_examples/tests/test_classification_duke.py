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

import shutil
import tempfile
import unittest
import os
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess

from fuse.utils.rand.seed import Seed
import fuse.utils.gpu as GPU

# if "DUKE_DATA_PATH" in os.environ:
#     from fuse_examples.imaging.classification.duke_breast_cancer.runner_duke import get_setting, run_train, run_infer, run_eval

# @unittest.skipIf("DUKE_DATA_PATH" not in os.environ, "define environment variable 'DUKE_DATA_PATH' to run this test")
@unittest.skipIf(True)
class ClassificationDukeTestCase(unittest.TestCase):
    def setUp(self):
        selected_positive = [1, 2, 3, 5, 6, 10, 12, 596, 900, 901]
        selected_negative = [4, 6, 7, 8, 11, 13, 14, 120, 902, 903]

        selected_sample_ids = [f"Breast_MRI_{ii:03d}" for ii in selected_positive + selected_negative]
        PATHS, TRAIN_COMMON_PARAMS, INFER_COMMON_PARAMS, EVAL_COMMON_PARAMS = get_setting(
            "default", selected_sample_ids=selected_sample_ids, num_epoch=2
        )

        self.root = tempfile.mkdtemp()

        self.paths = {
            "model_dir": os.path.join(self.root, "duke/model_dir"),
            "force_reset_model_dir": True,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
            "data_dir": PATHS["data_dir"],
            "cache_dir": os.path.join(self.root, "duke/cache_dir"),
            "data_split_filename": os.path.join(self.root, "split.pkl"),
            "inference_dir": os.path.join(self.root, "duke/infer_dir"),
            "eval_dir": os.path.join(self.root, "duke/analyze_dir"),
        }

        self.train_common_params = TRAIN_COMMON_PARAMS
        # self.train_common_params["manager.train_params"]["num_epochs"] = 2
        self.infer_common_params = INFER_COMMON_PARAMS

        self.analyze_common_params = EVAL_COMMON_PARAMS

    @run_in_subprocess(1200)
    def test_template(self):
        GPU.choose_and_enable_multiple_gpus(1)

        Seed.set_seed(0, False)  # previous test (in the pipeline) changed the deterministic behavior to True
        run_train(self.paths, self.train_common_params, reset_cache=True, audit_cache=False)
        run_infer(self.paths, self.infer_common_params, audit_cache=False)
        results = run_eval(self.paths, self.analyze_common_params)

        self.assertTrue("metrics.auc" in results)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)


if __name__ == "__main__":
    unittest.main()
