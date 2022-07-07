
import json
import os


from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool
from torch.utils.data.dataloader import DataLoader
from fuse.utils.ndict import NDict

from fuse.data import DatasetDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data import PipelineDefault, OpSampleAndRepeat, OpToTensor, OpRepeat
from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.data.ops.ops_common import OpLambda, OpZScoreNorm
from fuseimg.data.ops.aug.color import OpAugColor
from fuseimg.data.ops.aug.geometry import OpAugAffine2D, OpCrop3D, OpRotation3D, OpResizeTo
from fuseimg.data.ops.image_loader import OpLoadImage 
from fuseimg.data.ops.color import OpClip, OpToRange

import numpy as np
from fuse.data.utils.sample import get_sample_id
from typing import Hashable, List, Optional, Sequence, Tuple, Union
from functools import partial
import torch
from torch import Tensor
import pandas as pd
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault





class OpKnightSampleIDDecode(OpBase):
    '''
    decodes sample id into image and segmentation filename
    '''

    def __call__(self, sample_dict: NDict, test: bool = False) -> NDict:#, op_id: Optional[str]) -> NDict:
        '''
        
        '''

        sid = get_sample_id(sample_dict)
        sample_dict['data.input.case_id'] = sid
        img_filename_key = 'data.input.img_path'
        if test:
            sample_dict[img_filename_key] =   f'images/{sid}.nii.gz'
        else:
            sample_dict[img_filename_key] =   os.path.join(sid, 'imaging.nii.gz')

            seg_filename_key = 'data.gt.seg_path'
            sample_dict[seg_filename_key] = os.path.join(sid, 'aggregated_MAJ_seg.nii.gz')

        
        return sample_dict

class OpClinicalLoad(OpBase):
    def __init__(self, json_path: str):
        super().__init__()
        self.json_path = json_path

    def __call__(self, sample_dict: NDict, test: bool = False) -> NDict:
        cols = ['case_id', 'age_at_nephrectomy', 'body_mass_index', 'gender', 'comorbidities', \
                'smoking_history', 'radiographic_size', 'last_preop_egfr']
        
        if test:
            json_data = pd.read_json(os.path.join(self.json_path,'features.json'))[cols]
        else:
            cols += ["aua_risk_group"]
            json_data = pd.read_json(os.path.join(self.json_path,'knight.json'))[cols]
        
        sid = sample_dict['data.input.case_id']
        row = json_data[json_data["case_id"]==sid].to_dict("records")[0]

        row['gender'] = int(row['gender'].lower() == 'female') #female:1 | male:0
        row["comorbidities"] = int(any(x for x in row["comorbidities"].values())) #if has any comorbidity it is set to 1
        row['smoking_history'] = ['never_smoked','previous_smoker','current_smoker'].index(row['smoking_history'])
        if row['last_preop_egfr'] is None or row['last_preop_egfr']['value'] is None:
            row['last_preop_egfr'] = 77 # median value 
        elif row['last_preop_egfr']['value'] in ('>=90', '>90'):
            row['last_preop_egfr'] = 90
        else:
            row['last_preop_egfr'] = row['last_preop_egfr']['value']

        if row['radiographic_size'] is None:
            row['radiographic_size'] = 4.1 # this is the median value on the training set
        if not test:
            sample_dict["data.gt.gt_global.task_1_label"] = int(row["aua_risk_group"] in ['high_risk', 'very_high_risk'])
            sample_dict["data.gt.gt_global.task_2_label"] = ['benign','low_risk','intermediate_risk','high_risk', 'very_high_risk'].index(row["aua_risk_group"])

        sample_dict["data.input.clinical"] = row

        return sample_dict
        
class OpPrepare_Clinical(OpBase):

    def __call__(self, sample_dict: NDict) -> NDict:#, op_id: Optional[str]) -> NDict:, op_id: Optional[str]) -> NDict:
        age = sample_dict['data.input.clinical.age_at_nephrectomy']
        if age!=None and age > 0 and age < 120:
            age = np.array(age / 120.0).reshape(-1)
        else:
            age = np.array(-1.0).reshape(-1)
        
        bmi = sample_dict['data.input.clinical.body_mass_index']
        if bmi!=None and bmi > 10 and bmi < 100:
            bmi = np.array(bmi / 50.0).reshape(-1)
        else:
            bmi = np.array(-1.0).reshape(-1)

        radiographic_size = sample_dict['data.input.clinical.radiographic_size']
        if radiographic_size!=None and radiographic_size > 0 and radiographic_size < 50:
            radiographic_size = np.array(radiographic_size / 15.0).reshape(-1)
        else:
            radiographic_size = np.array(-1.0).reshape(-1)
        
        preop_egfr = sample_dict['data.input.clinical.last_preop_egfr']
        if preop_egfr!=None and preop_egfr > 0 and preop_egfr < 200:
            preop_egfr = np.array(preop_egfr / 90.0).reshape(-1)
        else:
            preop_egfr = np.array(-1.0).reshape(-1)
        # turn categorical features into one hot vectors
        gender = sample_dict['data.input.clinical.gender']
        gender_one_hot = np.zeros(len(GENDER_INDEX))
        if gender in GENDER_INDEX.values():
            gender_one_hot[gender] = 1

        comorbidities = sample_dict['data.input.clinical.comorbidities']
        comorbidities_one_hot = np.zeros(len(COMORBIDITIES_INDEX))
        if comorbidities in COMORBIDITIES_INDEX.values():
            comorbidities_one_hot[comorbidities] = 1
        
        smoking_history = sample_dict['data.input.clinical.smoking_history']
        smoking_history_one_hot = np.zeros(len(SMOKE_HISTORY_INDEX))
        if smoking_history in SMOKE_HISTORY_INDEX.values():
            smoking_history_one_hot[smoking_history] = 1
        

        clinical_encoding = np.concatenate((age, bmi, radiographic_size, preop_egfr, gender_one_hot, comorbidities_one_hot, smoking_history_one_hot), axis=0, dtype=np.float32)
        sample_dict["data.input.clinical.all"] = clinical_encoding
        return sample_dict

def knight_dataset(data_dir: str = 'data', cache_dir: str = 'cache', split: dict = None, \
        reset_cache: bool = False, \
        rand_gen = None, batch_size=8, resize_to=(110,256,256), task_num=1, \
        target_name='data.gt.gt_global.task_1_label', num_classes=2, only_labels=False):
    

    static_pipeline = PipelineDefault("static", [
        # decoding sample ID
        (OpKnightSampleIDDecode(), dict(test=('test' in split))), # will save image and seg path to "data.input.img_path", "data.gt.seg_path" and load json data
        (OpClinicalLoad(data_dir), dict(test=('test' in split))),
        # loading data
        (OpLoadImage(data_dir), dict(key_in="data.input.img_path", key_out="data.input.img", format="nib")),
        # (OpLoadImage(data_dir), dict(key_in="data.gt.seg_path", key_out="data.gt.seg", format="nib")),
        
        
        # fixed image normalization
        (OpClip(), dict(key="data.input.img", clip=(-62, 301))),
        (OpZScoreNorm(), dict(key="data.input.img", mean=104.0, std=75.3)), #kits normalization
        
        # transposing so the depth channel will be first
        (OpLambda(lambda x: np.moveaxis(x, -1, 0)), dict(key="data.input.img")), # convert image from shape [H, W, D] to shape [D, H, W] 
        (OpPrepare_Clinical(), dict()), #process clinical data

    ])

    val_dynamic_pipeline = PipelineDefault("dynamic", [
        (OpResizeTo(channels_first=False), dict(key="data.input.img", output_shape=resize_to)),
        # Numpy to tensor
        (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
        (OpToTensor(), dict(key="data.input.clinical.all")),

        # add channel dimension -> [C=1, D, H, W]
        (OpLambda(lambda x: x.unsqueeze(dim=0)), dict(key="data.input.img")),  
    ]) 

    train_dynamic_pipeline = PipelineDefault("dynamic", [
        (OpResizeTo(channels_first=False), dict(key="data.input.img", output_shape=resize_to)),
        # Numpy to tensor
        (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
        (OpToTensor(), dict(key="data.input.clinical.all")),

        (OpRandApply(OpSample(OpRotation3D()), 0.5) , dict(
            key="data.input.img",
            z_rot=Uniform(-5.0,5.0),
            x_rot=Uniform(-5.0,5.0),
            y_rot=Uniform(-5.0,5.0))
        ),
        # affine transformation per slice but with the same arguments
        (OpRandApply(OpSample(OpAugAffine2D()), 0.5) , dict(
            key="data.input.img",
            rotate=Uniform(-180.0,180.0),
            scale=Uniform(0.8, 1.2),
            flip=(RandBool(0.5), RandBool(0.5)),
            translate=(RandInt(-15, 15), RandInt(-15, 15))
        )),
        # add channel dimension -> [C=1, D, H, W]
        (OpLambda(lambda x: x.unsqueeze(dim=0)), dict(key="data.input.img")),  
    ])
       
    
    if 'train' in split:
        image_dir = data_dir
        json_filename = os.path.join(image_dir, 'knight.json')
    
    else: # split can contain BOTH 'train' and 'val', or JUST 'test'
        image_dir = os.path.join(data_dir, 'images')
        json_filepath = os.path.join(data_dir, 'features.json')
        
       # Create dataset
    if 'train' in split:
        train_cacher = SamplesCacher("train_cache", 
        static_pipeline,
        cache_dirs=[f"{cache_dir}/train"], restart_cache=reset_cache)

        train_dataset = DatasetDefault(sample_ids=split['train'],
        static_pipeline=static_pipeline,
        dynamic_pipeline=train_dynamic_pipeline,
        cacher=train_cacher)

        print(f'- Load and cache data:')
        train_dataset.create()
    
        print(f'- Load and cache data: Done')

        ## Create sampler
        print(f'- Create sampler:')
        sampler = BatchSamplerDefault(dataset=train_dataset,
                                        balanced_class_name=target_name,
                                        num_balanced_classes=num_classes,
                                        batch_size=batch_size,
                                        balanced_class_weights=[1.0/num_classes]*num_classes if task_num==2 else None)
                                                              

        print(f'- Create sampler: Done')

        ## Create dataloader
        train_dataloader = DataLoader(dataset=train_dataset,
                                    shuffle=False, drop_last=False,
                                    batch_sampler=sampler, collate_fn=CollateDefault(),
                                    num_workers=8, generator=rand_gen)
        print(f'Train Data: Done', {'attrs': 'bold'})

        #### Validation data
        print(f'Validation Data:', {'attrs': 'bold'})

        val_cacher = SamplesCacher("val_cache", 
            static_pipeline,
            cache_dirs=[f"{cache_dir}/val"], restart_cache=reset_cache)
        ## Create dataset
        validation_dataset = DatasetDefault(sample_ids=split['val'],
        static_pipeline=static_pipeline,
        dynamic_pipeline=val_dynamic_pipeline,
        cacher=val_cacher)

        print(f'- Load and cache data:')
        validation_dataset.create()
        print(f'- Load and cache data: Done')

        ## Create dataloader
        validation_dataloader = DataLoader(dataset=validation_dataset,
                                        shuffle=False,
                                        drop_last=False,
                                        batch_sampler=None,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        collate_fn=CollateDefault(),
                                        generator=rand_gen)
        print(f'Validation Data: Done', {'attrs': 'bold'})
        test_dataloader = test_dataset = None
    else: # test only
        #### Test data
        print(f'Test Data:', {'attrs': 'bold'})

        ## Create dataset
        test_dataset = DatasetDefault(sample_ids=split['test'],
        static_pipeline=static_pipeline,
        dynamic_pipeline=val_dynamic_pipeline,)

        print(f'- Load and cache data:')
        test_dataset.create()
        print(f'- Load and cache data: Done')

        ## Create dataloader
        test_dataloader = DataLoader(dataset=test_dataset,
                                        shuffle=False,
                                        drop_last=False,
                                        batch_sampler=None,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        collate_fn=CollateDefault(),
                                        generator=rand_gen)
        print(f'Test Data: Done', {'attrs': 'bold'})
        train_dataloader = train_dataset = validation_dataloader = validation_dataset = None
    return train_dataloader, validation_dataloader, test_dataloader, \
            train_dataset, validation_dataset, test_dataset


GENDER_INDEX = {
    'male': 0,
    'female': 1
}
COMORBIDITIES_INDEX = {
    'no comorbidities': 0,
    'comorbidities exist': 1
}
SMOKE_HISTORY_INDEX = {
    'never smoked': 0,
    'previous smoker': 1,
    'current smoker': 2
}
