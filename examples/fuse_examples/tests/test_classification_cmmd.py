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
# FIXME: data_package
from fuse_examples.imaging.classification.cmmd.runner import run_train, run_eval, run_infer
from fuse.utils import NDict
import unittest
import shutil
import tempfile
import os
from fuse.utils.gpu import choose_and_enable_multiple_gpus
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess
assert "CMMD_DATA_PATH" in os.environ, "Expecting environment variable CMMD_DATA_PATH to be set. Follow the instruction in example README file to download and set the path to the data"

# @unittest.skip("Not ready yet")
# TODO:
# 1. Get the path to data as an env variable
# 2. Consider reducing the number of samples
class ClassificationMGCmmdTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.cfg = NDict({'paths': 
            {'data_dir': os.environ["CMMD_DATA_PATH"],
             'model_dir':  os.path.join(self.root, 'model_new/InceptionResnetV2_2017_test'),
             'inference_dir': os.path.join(self.root,'model_new/infer_dir'),
             'eval_dir': os.path.join(self.root,'model_new/eval_dir'),
             'cache_dir': os.path.join(self.root,'examples/CMMD_cache_dir'),
             'data_misc_dir': os.path.join(self.root,'data_misc'),
             'data_split_filename': 'cmmd_split.pkl'},
            'train': {'force_reset_model_dir': True,
                      'target': 'classification',
                      'reset_cache': False,
                      'num_workers': 1, 
                      'num_folds': 5,
                      'train_folds': [0, 1, 2],
                      'validation_folds': [3],
                      'batch_size': 2,
                      'learning_rate': 0.0001,
                      'weight_decay': 0,
                      'resume_checkpoint_filename': None,
                      'manager_train_params': {'num_gpus': 1, 'device': 'cuda', 'num_epochs': 3, 'virtual_batch_size': 1, 'start_saving_epochs': 10, 'gap_between_saving_epochs': 100},
                      'manager_best_epoch_source': {'source': 'metrics.auc.macro_avg', 'optimization': 'max', 'on_equal_values': 'better'}},
            'infer': {'infer_filename': 'validation_set_infer.gz', 'infer_folds': [4], 'target': 'classification', 'checkpoint': 'best', 'num_workers': 1}})
        print(self.cfg)
        
        # Path to the stored dataset location
        # dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
        # download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
        # put on the folliwing in the main folder  - 
        # 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx 
        # 2. folder named CMMD which is the downloaded data folder

    @run_in_subprocess()
    def test_runner(self):
        # uncomment if you want to use specific gpus instead of automatically looking for free ones
        force_gpus = None  # [0]
        choose_and_enable_multiple_gpus(self.cfg["train.manager_train_params.num_gpus"], force_gpus=force_gpus)

        run_train(self.cfg["paths"] ,self.cfg["train"])
        run_infer(self.cfg["paths"] , self.cfg["infer"])
        results = run_eval(self.cfg["paths"] , self.cfg["infer"])

        self.assertTrue('metrics.auc' in results)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)
        
def main() -> None:
    unittest.main()
if __name__ == '__main__':
    main()

