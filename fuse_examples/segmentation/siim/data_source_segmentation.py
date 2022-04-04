import pandas as pd
from glob import glob
import random
import pickle
from typing import Sequence, Hashable, Union, Optional, List, Dict
from fuse.data.data_source.data_source_base import FuseDataSourceBase
from fuse.utils.utils_misc import autodetect_input_source


class FuseDataSourceSeg(FuseDataSourceBase):
    def __init__(self, 
                 image_source: str,
                 mask_source: Optional[str] = None, 
                 partition_file: Optional[str] = None,
                 train: bool = True,
                 val_split: float = 0.2,
                 override_partition: bool = True,
                 data_shuffle: bool = True
                 ):
        """
        Create DataSource
        :param input_source:       path to images
        :param partition_file:     Optional, name of a pickle file when no validation set is available
                                   If train = True, train/val indices are dumped into the file,
                                   If train = False, train/val indices are loaded
        :param train:              specifies if we are in training phase
        :param val_split:          validation proportion in case of splitting
        :param override_partition: specifies if the given partition file is filled with new train/val splits
        """


        # Extract entities
        # ----------------
        if partition_file is not None:
            if train:
                if override_partition:
                    train_fn = glob(image_source + '/*')
                    train_fn.sort()

                    masks_fn = glob(mask_source + '/*')
                    masks_fn.sort()

                    fn = list(zip(train_fn, masks_fn))
                    
                    if len(fn) == 0:
                        raise Exception('Error detecting input source in FuseDataSourceDefault')

                    if data_shuffle:
                        # random shuffle the file-list
                        random.shuffle(fn)

                    # split to train-validation -
                    n_train = int(len(fn) * (1-val_split))

                    train_fn = fn[:n_train]
                    val_fn = fn[n_train:]
                    splits = {'train': train_fn, 'val': val_fn}

                    with open(partition_file, "wb") as pickle_out:
                        pickle.dump(splits, pickle_out)
                    sample_descs = train_fn
                else:
                    # read from a previous train/test split to evaluate on the same partition
                    with open(partition_file, "rb") as splits:
                        repartition = pickle.load(splits)
                    sample_descs = repartition['train']
            else:
                with open(partition_file, "rb") as splits:
                    repartition = pickle.load(splits)
                sample_descs = repartition['val']
        else:
            for sample_id in input_df.iloc[:, 0]:
                sample_descs.append(sample_id)

        self.samples = sample_descs

        self.input_source = [image_source, mask_source]


    def get_samples_description(self):
        return self.samples

    def summary(self) -> str:
        summary_str = ''
        summary_str += 'FuseDataSourceSeg - %d samples\n' % len(self.samples)
        return summary_str