import os
import unittest
from testbook import testbook
from fuse.templates.walkthrough_template import NUM_GPUS
import fuse.utils.gpu as FuseUtilsGPU

class NotebookHelloWorldTestCase(unittest.TestCase):
    
    def test_notebook(self):
        NUM_GPUS = 1
        force_gpus = [0]
        FuseUtilsGPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)


        # Execute the whole notebook and save it as an object
        with testbook('fuse_examples/tutorials/hello_world/hello_world.ipynb', execute=True, timeout=600) as tb:

            test_result_acc = tb.ref("test_result_acc")
            assert(test_result_acc > 0.95)


if __name__ == '__main__':
    unittest.main()