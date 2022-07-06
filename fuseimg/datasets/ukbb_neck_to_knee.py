from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.color import OpNormalizeAgainstSelf
from fuseimg.data.ops.shape_ops import OpFlipBrightSideOnLeft2D , OpFindBiggestNonEmptyBbox2D, OpResizeAndPad2D
from fuse.data import PipelineDefault, OpToTensor
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D 
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_cast import OpToNumpy
from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
from functools import partial
from typing import Hashable, Optional, Sequence
import torch
import pandas as pd
import numpy as np
import pydicom
import io
import os
import multiprocessing
from pathlib import Path
from fuse.data.utils.sample import get_sample_id
import zipfile
# from fuseimg.data.ops import ops_mri
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool
import SimpleITK as sitk
import tempfile
import shutil
import skimage
import skimage.transform

from matplotlib import pyplot as plt
def dump(img, filename, slice):
    plt.imshow(img[slice,:,:], interpolation='nearest')
    plt.savefig(filename)
    return img

class OpUKBBSampleIDDecode(OpBase):
    '''
    decodes sample id into image and segmentation filename
    '''

    def __call__(self, sample_dict: NDict) -> NDict:
        '''
        
        '''
        sid = get_sample_id(sample_dict)
        
        img_filename_key = 'data.input.img_path'
        sample_dict[img_filename_key] =   sid

        return sample_dict

class OpLoadUKBBZip(OpBase):
    '''
    loads a zip and select a sequence and a station from it
    '''
    def __init__(self, dir_path: str, **kwargs):
        super().__init__(**kwargs)
        self._dir_path = dir_path

    def __call__(self, sample_dict: NDict, series : str , station : int, key_in:str, key_out: str, unique_id_out: str) -> NDict:
        '''
        
        '''
        scans = []
        zip_filename = os.path.join(self._dir_path,sample_dict[key_in])
        zip_file = zipfile.ZipFile(zip_filename)
        filenames_list = [f.filename for f in zip_file.infolist() if '.dcm' in f.filename]
        
        for dicom_file in filenames_list:
            
            with zip_file.open(dicom_file) as f:
                dcm = pydicom.read_file(io.BytesIO(f.read()))
                scans.append({'file': zip_filename.split("/")[-1], 'dcm_unique': dcm[0x0020000e].value, 'time':dcm[0x00080031].value, 'series': dcm[0x0008103e].value})
                
        dicom_tags = pd.DataFrame(scans)
        dicom_tags['n_slices'] = dicom_tags.groupby(dicom_tags.columns.to_list())['file'].transform('size')
        dicom_tags = dicom_tags.drop_duplicates()
        dicom_tags = dicom_tags.sort_values(by=['time'])
        station_list = []
        for i in range(6) :
            for j in range(4) :
                    station_list.append(i+1)
        dicom_tags['station'] = station_list
        dcm_unique = dicom_tags[dicom_tags['station'] == station][dicom_tags['series'] == series]['dcm_unique'].iloc[0]
        dirpath = tempfile.mkdtemp()
        # ... do stuff with dirpath
        for dicom_file in filenames_list:
            with zip_file.open(dicom_file) as f:
                if pydicom.read_file(io.BytesIO(f.read()))[0x0020000e].value  == dcm_unique :
                    zip_file.extract(dicom_file, path=dirpath)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dirpath)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        numpy_img = sitk.GetArrayFromImage(image)
        sample_dict[key_out] = numpy_img
        sample_dict[unique_id_out] = dcm_unique
        shutil.rmtree(dirpath)
        return sample_dict

class UKBB:
    """
    # dataset that contains MRI nech-to-knee  and metadata from UK patients
    # Path to the stored dataset location
    # put on the folliwing in the main folder  - 
    # 1. label.csv 
    # 2. folder named body-mri-data which is the downloaded data folder
    """
    # bump whenever the static pipeline modified
    CMMD_DATASET_VER = 0

    @staticmethod
    def static_pipeline(data_dir: str,data_source : pd.DataFrame, target: str) -> PipelineDefault:
        """
        Get suggested static pipeline (which will be cached), typically loading the data plus design choices that we won't experiment with.
        :param data_path: path to original kits21 data (can be downloaded by KITS21.download())
        """
        static_pipeline = PipelineDefault("cmmd_static", [
         # decoding sample ID
            # (OpUKBBSampleIDDecode(), dict()), # will save image and seg path to "data.input.img_path", "data.gt.seg_path"    
            (OpLoadUKBBZip(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", unique_id_out="data.ID", series="Dixon_BH_17s_W", station = 4)),
            (OpLambda(partial(skimage.transform.resize,
                                                            output_shape=(32, 256, 256),
                                                            mode='reflect',
                                                            anti_aliasing=True,
                                                            preserve_range=True)), dict(key="data.input.img")),
            (OpNormalizeAgainstSelf(), dict(key="data.input.img")),
            (OpLambda(partial(dump, filename="first.png", slice = 25)), dict(key="data.input.img")),
            (OpReadDataframe(data_source,
                    key_column="data.ID",key_name="data.ID", columns_to_extract=['patient_id','dcm_unique','is female'],
                    rename_columns={'dcm_unique' : 'data.ID' ,'patient_id' :"data.patientID", 'is female': "data.gt.classification" }), dict()),
            ])
        return static_pipeline

    # @staticmethod
    # def static_pipeline(root_path, select_series_func, with_rescale:Optional[bool]=True,
    #                     keep_stk_volumes:Optional[bool]=False, verbose:Optional[bool]=True) -> PipelineDefault:

    #     data_path = os.path.join(root_path, 'PROSTATEx')
    #     # metadata_path = os.path.join(root_path, 'metadata.csv')

    #     static_pipeline_steps = [
    #         # step 1: map sample_ids to
    #         (OpUKBBSampleIDDecode(data_path=data_path),
    #          dict(key_out='data.input.mri_path')),
    #         # step 2: read files info for the sequences
    #         # (ops_mri.OpExtractDicomsPerSeq(seq_ids=seq_ids, series_desc_2_sequence_map=series_desc_2_sequence_map,
    #         #                                    use_order_indicator=False),
    #         #  dict(key_in='data.input.mri_path',
    #         #       key_out_sequences='data.input.sequence_ids',
    #         #       key_out_path_prefix='data.input.path.',
    #         #       key_out_dicoms_prefix='data.input.dicoms.',
    #         #       key_out_series_num_prefix='data.input.series_num.')
    #         #  ),
    #         # step 3: Load STK volumes of MRI sequences
    #         (ops_mri.OpLoadDicomAsStkVol(reverse_order=[True,True,True,True], is_file=False),
    #          dict(key_in_seq_ids='data.input.sequence_ids',
    #               key_in_path_prefix='data.input.path.',
    #               key_in_dicoms_prefix='data.input.dicoms.',
    #               key_out_prefix='data.input.volumes.')),
    #         # # step 4: group DCE sequnces into DCE_mix
    #         # (ops_mri.OpGroupDCESequences(),
    #         #  dict(key_sequence_ids='data.input.sequence_ids',
    #         #       key_path_prefix='data.input.path.',
    #         #       key_series_num_prefix='data.input.series_num.',
    #         #       key_volumes_prefix='data.input.volumes.')),

    #         # step 5: select single volume from b_mix/T2 sequence
    #         (ops_mri.OpSelectVolumes(get_indexes_func=select_series_func, delete_input_volumes=True),
    #          dict(key_in_sequence_ids='data.input.sequence_ids',
    #               key_in_path_prefix='data.input.path.',
    #               key_in_volumes_prefix='data.input.volumes.',
    #               key_out_paths='data.input.selected_paths',
    #               key_out_volumes='data.input.selected_volumes')),

    #         # step 6: set reference volume to be first and register other volumes with respect to it
    #         (ops_mri.OpResampleStkVolsBasedRef(reference_inx=0, interpolation='bspline'),
    #          dict(key='data.input.selected_volumes')),

    #         # step 7: create a single 4D volume from all the sequences (4th channel is the sequence)
    #         (ops_mri.OpStackList4DStk(delete_input_volumes=True), dict(key_in='data.input.selected_volumes',
    #                                                                        key_out_volume4d='data.input.volume4D',
    #                                                                        key_out_ref_volume='data.input.ref_volume')),

    #         ]
    #     if with_rescale:
    #         # step 8:
    #         static_pipeline_steps += [(ops_mri.OpRescale4DStk(), dict(key='data.input.volume4D'))]

    #     if keep_stk_volumes:
    #         static_pipeline_steps += [
    #             # step 11: move to ndarray - to allow quick saving
    #             (ops_mri.OpStk2Dict(),
    #              dict(keys=['data.input.volume4D','data.input.ref_volume']))
    #             ]
    #     static_pipeline = PipelineDefault("static", static_pipeline_steps, verbose=verbose)

    #     return static_pipeline

    @staticmethod
    def dynamic_pipeline(train: bool = False):
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations. 
        :param train : True iff we request dataset for train purpouse
        """
        dynamic_pipeline = PipelineDefault("cmmd_dynamic", [
            (OpToTensor(), dict(key="data.input.img",dtype=torch.float32)),
            (OpLambda(partial(torch.unsqueeze, dim=0)), dict(key="data.input.img")) ])
        if train:
            dynamic_pipeline.extend([
                (OpRandApply(OpSample(OpAugAffine2D()), 0.5), dict(
                                key="data.input.img",
                                rotate=Uniform(-30.0,30.0),        
                                scale=Uniform(0.9, 1.1),
                                flip=(RandBool(0.3), RandBool(0.5)),
                                translate=(RandInt(-10, 10), RandInt(-10, 10))
                            )),
                (OpRandApply(OpSample(OpAugColor()), 0.5), dict(
                            key="data.input.img",
                            gamma=Uniform(0.9, 1.1), 
                            contrast=Uniform(0.85, 1.15),
                            mul =  Uniform(0.95, 1.05),
                            add=Uniform(-0.06, 0.06)
                        )),
                
                
            ])
        return dynamic_pipeline


    
    def get_dicom_data_df(gt_file_path: str, data_dir: str, data_misc_dir:str, target: str,sample_ids : Sequence = None) -> str:
        """
        Creates a csv file that contains label for each image ( instead of patient as in dataset given file)
        by reading metadata ( breast side and view ) from the dicom files and merging it with the input csv
        If the csv already exists , it will skip the creation proccess
        :param gt_file_path                 path to ground trouth file
        :param data_dir                     dataset root path
        :param data_misc_dir                path to save misc files to be used later
        :param sample_ids                      list of ids to scan in data_dir
        :return: the new csv file path
        :return: sample ids of used images
        """


    
        combined_file_path = os.path.join(data_misc_dir, 'files_combined.csv')
        if os.path.isfile(combined_file_path):
            print("Found ground truth file:",combined_file_path )
            merged_clinical_data = pd.read_csv(combined_file_path)
            merged_clinical_data = merged_clinical_data[merged_clinical_data['file'].isin(sample_ids)]
            all_sample_ids = merged_clinical_data['file'].to_list()
            return merged_clinical_data, all_sample_ids
        print("Did not find exising ground truth file!")
        Path(data_misc_dir).mkdir(parents=True, exist_ok=True)
        if sample_ids != None :
            zip_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file in sample_ids]
        else:
            zip_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.zip' in file]
        with multiprocessing.Pool(64) as pool:
            dfs = [x for x in pool.imap(create_df_from_zip, zip_files) if x is not None]
        df = pd.concat(dfs)
        if gt_file_path is not None:
            gt_file = pd.read_csv(gt_file_path)
            df = pd.merge(df, gt_file, how='inner', on=['file'])
        else:
            print("Did not merge with ground truth file as it was None")
        df.to_csv(combined_file_path)
        all_sample_ids = df['file'].to_list()
        return df, all_sample_ids
    
    @staticmethod
    def dataset(
                data_dir: str,
                data_misc_dir : str,
                target: str,
                cache_dir : str = None,
                reset_cache : bool = True,
                num_workers:int = 10,
                sample_ids: Optional[Sequence[Hashable]] = None,
                train: bool = False,
                gt_file_path: str = None,) :
        """
        Creates Fuse Dataset single object (either for training, validation and test or user defined set)
        
        :param data_dir:                    dataset root path
        :param data_misc_dir                path to save misc files to be used later
        :param target                       target name used from the ground truth dataframe
        :param cache_dir:                   Optional, name of the cache folder
        :param reset_cache:                 Optional,specifies if we want to clear the cache first
        :param num_workers: number of processes used for caching 
        :param sample_ids: dataset including the specified sample_ids or None for all the samples. sample_id is case_{id:05d} (for example case_00001 or case_00100).
        :param train: True if used for training  - adds augmentation operations to the pipeline
        :param gt_file_path                 path to ground trouth file
        :return: DatasetDefault object
        """
        input_source_gt , all_sample_ids= UKBB.get_dicom_data_df(gt_file_path, data_dir, data_misc_dir, target,sample_ids =sample_ids)
        target = 'is female'
        if sample_ids is None:
            sample_ids = all_sample_ids
            
        static_pipeline = UKBB.static_pipeline(data_dir,input_source_gt, target)
        dynamic_pipeline = UKBB.dynamic_pipeline(train=train)
                                
        cacher = SamplesCacher(f'cmmd_cache_ver', 
            static_pipeline,
            cache_dirs=[cache_dir],
            restart_cache=reset_cache,
            audit_first_sample=False, audit_rate=None,
            workers=num_workers)   
        
        my_dataset = DatasetDefault(sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,            
        )

        my_dataset.create()
        return my_dataset
def create_df_from_zip(file):
        scans = []
        zip_file = zipfile.ZipFile(file)
        filenames_list = [f.filename for f in zip_file.infolist() if '.dcm' in f.filename]
        for dicom_file in filenames_list:
            with zip_file.open(dicom_file) as f:
                dcm = pydicom.read_file(io.BytesIO(f.read()))
                scans.append({'file': file.split("/")[-1], 'dcm_unique': dcm[0x0020000e].value, 'time':dcm[0x00080031].value, 'series': dcm[0x0008103e].value,
                            'sex': dcm[0x00100040].value, 'birthday': dcm[0x00100030].value, 'age': dcm[0x00101010].value, 'size': dcm[0x00101020].value, 'weight': dcm[0x00101030].value})
        dicom_tags = pd.DataFrame(scans)
        dicom_tags['n_slices'] = dicom_tags.groupby(dicom_tags.columns.to_list())['file'].transform('size')
        dicom_tags = dicom_tags.drop_duplicates()
        dicom_tags = dicom_tags.sort_values(by=['time'])
        if len(dicom_tags) != 24:
            print(file, "has missing/extra sequences ",len(dicom_tags),"instead of 24")
            return None
        station_list = []
        for i in range(6) :
            for j in range(4) :
                    station_list.append(i+1)
        dicom_tags['station'] = station_list
        return dicom_tags