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
import multiprocessing
from fuse.utils import NDict
import unittest
import shutil
import tempfile
import os
from fuse.utils.gpu import choose_and_enable_multiple_gpus
from fuse.utils.multiprocessing.run_multiprocessed import run_in_subprocess

        
# os env variable CMMD_DATA_PATH is a path to the stored dataset location
# dataset should be download from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508
# download requires NBIA data retriever https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
# put on the folliwing in the main folder  - 
# 1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx 
# 2. folder named CMMD which is the downloaded data folder
if "CMMD_DATA_PATH" in os.environ:
    from fuse_examples.imaging.classification.cmmd.runner import run_train, run_eval, run_infer

@unittest.skipIf("CMMD_DATA_PATH" not in os.environ, "define environment variable 'CMMD_DATA_PATH' to run this test")
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
              'run': {'running_modes': ['train', 'infer', 'eval']},
              'train': 
            {
             'target': 'classification', 
             'reset_cache': False, 
             'num_workers': 10, 
             'num_folds': 5, 
             'train_folds': [0, 1, 2], 
             'validation_folds': [3], 
             'batch_size': 2, 'learning_rate': 0.0001, 
             'weight_decay': 0, 
             'resume_checkpoint_filename': None, 
             'trainer': {'accelerator': 'gpu', 'devices': 1, 'num_epochs': 2, 'ckpt_path': None}}, 
            'infer': {'infer_filename': 'validation_set_infer.gz', 'checkpoint' : 'best_epoch.ckpt' , 'infer_folds': [4], 'target': 'classification', 'num_workers': 10}})
        
        print(self.cfg)

    # @run_in_subprocess()
    def test_runner(self):
        # uncomment if you want to use specific gpus instead of automatically looking for free ones
        force_gpus = None  # [0]
        choose_and_enable_multiple_gpus(self.cfg["train.trainer.devices"], force_gpus=force_gpus)

        run_train(self.cfg["paths"] ,self.cfg["train"])
        run_infer(self.cfg["train"], self.cfg["paths"] , self.cfg["infer"])
        results = run_eval(self.cfg["paths"] , self.cfg["infer"])

        self.assertTrue('metrics.auc' in results)

    def tearDown(self):
        # Delete temporary directories
        shutil.rmtree(self.root)
        
def main() -> None:

    unittest.main()
if __name__ == '__main__':
    main()

