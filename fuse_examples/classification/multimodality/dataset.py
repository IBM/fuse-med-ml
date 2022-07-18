from typing import Callable, Optional, Tuple, Any, Iterable
import logging
import pandas as pd
from typing import List

from fuse.data.visualizer.visualizer_default import FuseVisualizerDefault
from fuse.data.augmentor.augmentor_default import FuseAugmentorDefault
from fuse.data.augmentor.augmentor_toolbox import aug_op_color, aug_op_gaussian, aug_op_affine
from fuse.data.dataset.dataset_default import FuseDatasetDefault
from fuse.data.dataset.dataset_generator import FuseDatasetGenerator
from fuse.data.data_source.data_source_default import FuseDataSourceDefault
from fuse.data.processor.processor_base import FuseProcessorBase
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool

from fuse_examples.classification.multimodality.input_processor import ImagingTabularProcessor



def imaging_augmentation()-> Iterable[Any]:
    """
    :return: augmentation_pipeline iterator
    """
    augmentation_pipeline = [
        [
            ('data.image',),
            aug_op_affine,
            {'rotate': Uniform(-30.0, 30.0), 'translate': (RandInt(-10, 10), RandInt(-10, 10)),
             'flip': (RandBool(0.3), RandBool(0.3)), 'scale': Uniform(0.9, 1.1)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.image',),
            aug_op_color,
            {'add': Uniform(-0.06, 0.06), 'mul': Uniform(0.95, 1.05), 'gamma': Uniform(0.9, 1.1),
             'contrast': Uniform(0.85, 1.15)},
            {'apply': RandBool(0.5)}
        ],
        [
            ('data.image',),
            aug_op_gaussian,
            {'std': 0.03},
            {'apply': RandBool(0.5)}
        ],
    ]

    # Create data augmentation (optional)
    augmentor = FuseAugmentorDefault(augmentation_pipeline=augmentation_pipeline)

    return augmentor


# def tabular_dataset(tabular_processor,df,tabular_features,sample_key):
#
#
#     tabular_features.remove(sample_key)
#     tabular_processor = tabular_processor(data=df,
#                                           sample_desc_column=sample_key,
#                                           columns_to_extract=tabular_features + [sample_key],
#                                           columns_to_tensor=tabular_features)
#     return tabular_processor


def imaging_tabular_dataset(data_split: List[pd.DataFrame],
                            imaging_processor: FuseProcessorBase,
                            tabular_processor: FuseProcessorBase,
                            label_key:str,
                            img_key:str,
                            sample_key: str,
                            tabular_features_lst: list,
                            cache_dir: str = 'cache',
                            reset_cache: bool = False,
                            post_cache_processing_func: Optional[Callable] = None) -> Tuple[FuseDatasetDefault, FuseDatasetDefault]:

    """
    Creates Fuse Dataset object for training, validation and test
    :param data_split:                  A list of train, validation and test dataframes
    :param imaging_processor:           Imaging data generator
    :param tabular_processor:           Tabular data generator
    :param label_key                    Name of label to use from dataframe
    :param img_key                      Name of image path column
    :param sample_key                   Name of sample id
    :param tabular_features_lst         a list of tabular keys to use
    :param cache_dir:                   Optional, name of the cache folder
    :param reset_cache:                 Optional,specifies if we want to clear the cache first
    :param post_cache_processing_func:  Optional, function run post cache processing
    :return: training, validation and test FuseDatasetDefault objects
    """

    lgr = logging.getLogger('Fuse')

    if isinstance(data_split,list):
        df_train = data_split[0]
        if len(data_split)>1:
            df_val = data_split[1]
        if len(data_split)>2:
            df_test = data_split[2]
        else:
            raise Exception(f'current version supports train/val/test data division')

    #----------------------------------------------
    # -----Datasource
    train_data_source = FuseDataSourceDefault(input_source=df_train)
    validation_data_source = FuseDataSourceDefault(input_source=df_val)
    test_data_source = FuseDataSourceDefault(input_source=df_test)

    # ----------------------------------------------

    tabular_features_lst.remove(sample_key)
    # tabular_processor = tabular_processor(data=df,
    #                                       sample_desc_column=sample_key,
    #                                       columns_to_extract=tabular_features_lst + [sample_key],
    #                                       columns_to_tensor=tabular_features_lst)

    # -----Data-processors
    img_clinical_processor_train = ImagingTabularProcessor(data=df_train,
                                                           label=label_key,
                                                           img_key = img_key,
                                                           image_processor=imaging_processor(''),
                                                           tabular_processor=tabular_processor(data=df_train,
                                                                              sample_desc_column=sample_key,
                                                                              columns_to_extract=tabular_features_lst + [sample_key],
                                                                              columns_to_tensor=tabular_features_lst)
                                                           # tabular_dataset(tabular_processor,df_train,tabular_features_lst.copy(),sample_key)
                                                           )

    img_clinical_processor_val = ImagingTabularProcessor(data=df_val,
                                                         label=label_key,
                                                         img_key=img_key,
                                                         image_processor=imaging_processor(''),
                                                         tabular_processor=tabular_processor(data=df_val,
                                                                              sample_desc_column=sample_key,
                                                                              columns_to_extract=tabular_features_lst + [sample_key],
                                                                              columns_to_tensor=tabular_features_lst)
                                                         # tabular_dataset(tabular_processor,df_val,tabular_features_lst.copy(),sample_key)
                                                         )

    img_clinical_processor_test = ImagingTabularProcessor(data=df_test,
                                                          label=label_key,
                                                          img_key=img_key,
                                                          image_processor=imaging_processor(''),
                                                          tabular_processor=tabular_processor(data=df_test,
                                                                            sample_desc_column=sample_key,
                                                                            columns_to_extract=tabular_features_lst + [sample_key],
                                                                            columns_to_tensor=tabular_features_lst)
                                                          # tabular_dataset(tabular_processor,df_test,tabular_features_lst.copy(),sample_key)
                                                          )



    visualiser = FuseVisualizerDefault(image_name='data.image', label_name='data.gt')


    # ----------------------------------------------
    # ------ Dataset
    train_dataset = FuseDatasetGenerator(cache_dest=cache_dir,
                                       data_source=train_data_source,
                                       processor=img_clinical_processor_train,
                                       augmentor=imaging_augmentation(),
                                       visualizer=visualiser,
                                       post_processing_func=post_cache_processing_func,)


    validation_dataset = FuseDatasetGenerator(cache_dest=cache_dir,
                                       data_source=validation_data_source,
                                       processor=img_clinical_processor_val,
                                       augmentor=None,
                                       visualizer=visualiser,
                                       post_processing_func=post_cache_processing_func,)

    test_dataset = FuseDatasetGenerator(cache_dest=cache_dir,
                                       data_source=test_data_source,
                                       processor=img_clinical_processor_test,
                                       augmentor=None,
                                       visualizer=visualiser,
                                       post_processing_func=post_cache_processing_func,)

    # ----------------------------------------------
    # ------ Cache
    # create cache
    train_dataset.create(reset_cache=reset_cache)  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
    validation_dataset.create()  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading
    test_dataset.create()  # use ThreadPool to create this dataset, to avoid cv2 problems in multithreading

    lgr.info(f'- Load and cache data:')

    lgr.info(f'- Load and cache data: Done')

    return train_dataset, validation_dataset, test_dataset


