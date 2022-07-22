import numpy as np
import glob
import os
import radiomics
from typing import Hashable, Optional, Sequence

import fuse.data.ops.ops_common
# import fuseimg.data.ops.ops_common_imaging
from fuse.data.ops import ops_common
from functools import partial

from fuseimg.data.ops.aug import geometry, geometry3d
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform

import pandas as pd

from fuse.data import DatasetDefault
from fuse.data import PipelineDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.op_base import OpReversibleBase, OpBase
from fuse.data.ops import ops_read, ops_cast
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuseimg.data.ops import ops_mri

import torch

from fuseimg.datasets.duke_label_type import DukeLabelType


def get_selected_series_index(sample_id, seq_id):
    patient_id = sample_id[0]
    if patient_id in ["Breast_MRI_120", "Breast_MRI_596"]:
        map = {"DCE_mix": [2], "MASK": [0]}
    else:
        map = {"DCE_mix": [1], "MASK": [0]}
    return map[seq_id]


class Duke:
    DUKE_DATASET_VER = 0

    @staticmethod
    def dataset(
        label_type: Optional[DukeLabelType] = None,
        train: Optional[int] = False,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        select_series_func=get_selected_series_index,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        verbose: Optional[bool] = True,
        cache_kwargs: Optional[dict] = None,
    ) -> DatasetDefault:

        """
        :param label_type: type of label to use
        :param cache_dir: path to store the cache of the static pipeline
        :param data_dir: path to the original data
        :param select_series_func: which series to select for DCE_mix sequences
        :param reset_cache:
        :param num_workers:  number of processes used for caching
        :param sample_ids: list of selected patient_ids for the dataset
        :return:
        """

        if sample_ids is None:
            sample_ids = Duke.sample_ids()

        static_pipeline = Duke.static_pipeline(
            data_dir=data_dir, select_series_func=select_series_func, verbose=verbose
        )
        dynamic_pipeline = Duke.dynamic_pipeline(data_dir=data_dir, train=train, label_type=label_type, verbose=verbose)

        if cache_dir is None:
            cacher = None
        else:
            if cache_kwargs is None:
                cache_kwargs = {}
            cacher = SamplesCacher(
                f"duke_cache_ver{Duke.DUKE_DATASET_VER}",
                static_pipeline,
                [cache_dir],
                restart_cache=reset_cache,
                workers=num_workers,
                ignore_nan_inequality=True,
                **cache_kwargs,
            )

        my_dataset = DatasetDefault(
            sample_ids=sample_ids, static_pipeline=static_pipeline, dynamic_pipeline=dynamic_pipeline, cacher=cacher
        )
        my_dataset.create()
        return my_dataset

    @staticmethod
    def sample_ids():
        return [f"Breast_MRI_{i:03d}" for i in range(1, 923)]

    @staticmethod
    def static_pipeline(
        select_series_func,
        data_dir=None,
        with_rescale: Optional[bool] = True,
        output_stk_volumes: Optional[bool] = False,
        output_patch_volumes: Optional[bool] = True,
        verbose: Optional[bool] = True,
        duke_patch_annotations_df: Optional[pd.DataFrame] = None,
        name_suffix="",
    ) -> PipelineDefault:

        data_dir = Duke.get_data_dir_from_environment_variable() if data_dir is None else data_dir
        mri_dir = os.path.join(data_dir, "manifest-1607053360376")
        mri_dir2 = os.path.join(mri_dir, "Duke-Breast-Cancer-MRI")
        metadata_path = os.path.join(mri_dir, "metadata.csv")

        series_desc_2_sequence_map = get_series_desc_2_sequence_mapping(metadata_path)
        seq_ids = ["DCE_mix_ph1", "DCE_mix_ph2", "DCE_mix_ph3", "DCE_mix_ph4", "DCE_mix", "DCE_mix_ph"]

        static_pipeline_steps = [
            # step 1: map sample_ids to
            (OpDukeSampleIDDecode(data_path=mri_dir2), dict(key_out="data.input.mri_path")),
            # step 2: read sequences
            (
                ops_mri.OpExtractDicomsPerSeq(
                    seq_ids=seq_ids, series_desc_2_sequence_map=series_desc_2_sequence_map, use_order_indicator=False
                ),
                dict(
                    key_in="data.input.mri_path",
                    key_out_seq_ids="data.input.seq_ids",
                    key_out_sequence_prefix="data.input.sequence.",
                ),
            ),
            # step 3: Load STK volumes of MRI sequences
            (
                ops_mri.OpLoadDicomAsStkVol(),
                dict(key_in_seq_ids="data.input.seq_ids", key_sequence_prefix="data.input.sequence."),
            ),
            # step 4: group DCE sequences into DCE_mix
            (
                ops_mri.OpGroupDCESequences(),
                dict(key_seq_ids="data.input.seq_ids", key_sequence_prefix="data.input.sequence."),
            ),
            # step 5: select single volume from DCE_mix sequence
            (
                ops_mri.OpSelectVolumes(
                    get_indexes_func=select_series_func, selected_seq_ids=["DCE_mix"], delete_input_volumes=True
                ),
                dict(
                    key_in_seq_ids="data.input.seq_ids",
                    key_in_sequence_prefix="data.input.sequence.",
                    key_out_volumes="data.input.selected_volumes",
                    key_out_volumes_info="data.input.selected_volumes_info",
                ),
            ),
            # step 6: set first volume to be the reference volume and register other volumes with respect to it
            (
                ops_mri.OpResampleStkVolsBasedRef(reference_inx=0, interpolation="bspline"),
                dict(key="data.input.selected_volumes"),
            ),
            # step 7: create a single 4D volume from all the sequences (4th channel is the sequence)
            (
                ops_mri.OpStackList4DStk(delete_input_volumes=True),
                dict(
                    key_in="data.input.selected_volumes",
                    key_out_volume4d="data.input.volume4D",
                    key_out_ref_volume="data.input.ref_volume",
                ),
            ),
        ]
        if with_rescale:
            # step 8:
            static_pipeline_steps += [(ops_mri.OpRescale4DStk(), dict(key="data.input.volume4D"))]

        # step 9: read raw annotations - will be used for labels, features, and also for creating lesion properties
        static_pipeline_steps += [
            (
                ops_read.OpReadDataframe(data=get_duke_raw_annotations_df(data_dir), key_column="Patient ID"),
                dict(key_out_group="data.input.annotations"),
            )
        ]

        # step 10: add lesion properties for each sample id
        if output_patch_volumes:
            if duke_patch_annotations_df is not None:
                # read previousy computed patch annotations
                static_pipeline_steps += [
                    (
                        ops_read.OpReadDataframe(data=duke_patch_annotations_df, key_column="Patient ID"),
                        dict(key_out_group="data.input.patch_annotations"),
                    ),
                ]
            else:
                # generate patch annotations
                static_pipeline_steps += [
                    # add lesion features
                    (
                        ops_mri.OpExtractPatchAnotations(),
                        dict(
                            key_in_ref_volume="data.input.ref_volume",
                            key_in_annotations="data.input.annotations",
                            key_out="data.input.patch_annotations",
                        ),
                    )
                ]

            static_pipeline_steps += [
                # step 11: generate a mask from the lesion BB and append as a new (last) channel
                (
                    ops_mri.OpAddMaskFromBoundingBoxAsLastChannel(name_suffix=name_suffix),
                    dict(
                        key_volume4D="data.input.volume4D",
                        key_in_ref_volume="data.input.ref_volume",
                        key_in_patch_annotations="data.input.patch_annotations",
                    ),
                ),
                # step 12: create patch volumes using the mask channel: (i) fixed size (original scale) around center of annotatins (orig), and (ii) entire annotations
                (
                    ops_mri.OpCreatePatchVolumes(
                        lsn_shape=(9, 100, 100),
                        lsn_spacing=(1, 0.5, 0.5),
                        crop_based_annotation=True,
                        name_suffix=name_suffix,
                        delete_input_volumes=not output_stk_volumes,
                    ),
                    dict(
                        key_in_volume4D="data.input.volume4D",
                        key_in_ref_volume="data.input.ref_volume",
                        key_in_patch_annotations="data.input.patch_annotations",
                        key_out="data.input.patch_volume",
                    ),
                ),
            ]
        if output_stk_volumes:
            static_pipeline_steps += [
                # step 13: move STK volumes to ndarrays - to allow quick saving to disk
                (ops_mri.OpStk2Dict(), dict(keys=["data.input.volume4D", "data.input.ref_volume"]))
            ]
        static_pipeline = PipelineDefault("static", static_pipeline_steps, verbose=verbose)

        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        data_dir: Optional[str] = None,
        label_type: Optional[DukeLabelType] = None,
        train: Optional[bool] = False,
        num_channels: Optional[int] = 1,
        verbose: Optional[bool] = True,
        use_entire_lesion_volume: Optional[bool] = True,
        add_clinical_features: Optional[bool] = False,
    ):
        assert use_entire_lesion_volume
        volume_key = "data.input.patch_volume" if use_entire_lesion_volume else "data.input.patch_volume_orig"

        def delete_last_channel_in_volume(sample_dict: NDict):
            vol = sample_dict[volume_key]
            vol = vol[:-1]  # remove last channel
            sample_dict[volume_key] = vol
            return sample_dict

        dynamic_steps = [
            # step 1: delete the mask channel
            (fuse.data.ops.ops_common.OpLambda(func=delete_last_channel_in_volume), dict(key=None)),
            # step 2: turn volume to tensor
            (ops_cast.OpToTensor(), dict(key=volume_key, dtype=torch.float32)),
        ]
        if train:
            # step 3: augmentations
            dynamic_steps += [
                # step 3.1. 3D rotation
                # [
                #     ('data.input',),
                #     rotation_in_3d,
                #     {'z_rot': Uniform(-5.0, 5.0), 'y_rot': Uniform(-5.0, 5.0), 'x_rot': Uniform(-5.0, 5.0)},
                #     {'apply': RandBool(0.5)}
                # ],
                (
                    OpRandApply(OpSample(geometry3d.OpRotation3D()), 0.5),
                    dict(
                        key="data.input.patch_volume",
                        ax1_rot=Uniform(-5.0, 5.0),
                        ax2_rot=Uniform(-5.0, 5.0),
                        ax3_rot=Uniform(-5.0, 5.0),
                    ),
                ),
                # step 3.2.1 3D => 2D
                # [
                #     ('data.input',),
                #     squeeze_3d_to_2d,
                #     {'axis_squeeze': 'z'},
                #     {}
                # ],
                (geometry.OpAugSqueeze3Dto2D(), dict(key="data.input.patch_volume", axis_squeeze=1)),
                # step 3.2.2 2D affine transformation
                # [
                #     ('data.input',),
                #     aug_op_affine,
                #     {'rotate': Uniform(0, 360.0),
                #      'translate': (RandInt(-4, 4), RandInt(-4, 4)),
                #      'flip': (RandBool(0.5), RandBool(0.5)),
                #      'scale': Uniform(0.9, 1.1),
                #      },
                #     {'apply': RandBool(0.5)}
                # ],
                (
                    OpRandApply(OpSample(geometry.OpAugAffine2D()), 0.5),
                    dict(
                        key="data.input.patch_volume",
                        rotate=Uniform(0, 360.0),
                        scale=Uniform(0.9, 1.1),
                        flip=(RandBool(0.5), RandBool(0.5)),
                        translate=(RandInt(-4, 4), RandInt(-4, 4)),
                    ),
                ),
                # step 3.2.3 2D => 3D
                # [
                #     ('data.input',),
                #     unsqueeze_2d_to_3d,
                #     {'channels': num_channels, 'axis_squeeze': 'z'},
                #     {}
                # ],
                (
                    geometry.OpAugUnsqueeze3DFrom2D(),
                    dict(
                        key="data.input.patch_volume",
                        axis_squeeze=1,
                        channels=num_channels,
                    ),
                ),
            ]

        keys_2_keep = [volume_key]
        key_ground_truth = "data.ground_truth"
        if add_clinical_features or (label_type is not None):
            data_dir = Duke.get_data_dir_from_environment_variable() if data_dir is None else data_dir

            # step 4 (optional): read clinical data
            dynamic_steps += [
                (
                    ops_read.OpReadDataframe(
                        data=get_duke_clinical_data_df(data_dir), key_column="Patient Information:Patient ID"
                    ),
                    dict(key_out_group="data.input.clinical_data"),
                )
            ]
            if label_type is not None:
                # dynamic_steps.append((OpAddDukeLabelAndClinicalFeatures(label_type=label_type),
                #                 dict(key_in='data.input.clinical_data',  key_out_gt=key_ground_truth,
                #                      key_out_clinical_features='data.clinical_features')))
                #
                # step 5: add ground truth label
                dynamic_steps.append(
                    (
                        ops_common.OpLambda(func=label_type.get_value),
                        dict(key="data.input.clinical_data", key_out=key_ground_truth),
                    )
                )

                # step 6: remove entries with Nan labels
                dynamic_steps.append(
                    (
                        ops_common.OpLambda(func=partial(remove_entries_with_nan_label, key=key_ground_truth)),
                        dict(key=None),
                    )
                )
                keys_2_keep.append(key_ground_truth)

            if add_clinical_features:
                key_clinical_features = "data.clinical_features"
                dynamic_steps.append(
                    (
                        ops_common.OpLambda(func=label_type.select_features),
                        dict(key="data.input.clinical_data", key_out=key_clinical_features),
                    )
                )
                keys_2_keep.append(key_clinical_features)

            for key in keys_2_keep:
                if key == volume_key:
                    continue
                dtype = torch.int64 if key == key_ground_truth else torch.float32
                dynamic_steps += [(ops_cast.OpToTensor(), dict(key=key, dtype=dtype))]

        dynamic_steps.append((ops_common.OpKeepKeypaths(), dict(keep_keypaths=keys_2_keep)))
        dynamic_pipeline = PipelineDefault("dynamic", dynamic_steps, verbose=verbose)

        return dynamic_pipeline

    @staticmethod
    def get_data_dir_from_environment_variable():
        return os.environ["DUKE_DATA_PATH"]


class OpDukeSampleIDDecode(OpReversibleBase):
    """
    decodes sample id into path of MRI images
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path

    def __call__(self, sample_dict: NDict, key_out: str, op_id: Optional[str]) -> NDict:
        sid = get_sample_id(sample_dict)

        sample_dict[key_out] = get_sample_path(self._data_path, sid)

        return sample_dict

    def reverse(self, sample_dict: dict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return sample_dict


class OpAddDukeLabelAndClinicalFeatures(OpBase):
    """
    decodes sample id into path of MRI images
    """

    def __init__(self, label_type: DukeLabelType, is_concat_features_to_input: Optional[bool] = False, **kwargs):
        super().__init__(**kwargs)
        self._label_type = label_type
        self._is_concat_features_to_input = is_concat_features_to_input

    def __call__(self, sample_dict: NDict, key_in: str, key_out_gt: str, key_out_clinical_features: str) -> NDict:
        clinical_features = sample_dict[key_in]
        label_val = self._label_type.get_value(clinical_features)
        if np.isnan(label_val):
            return None  # should filter example (instead of sample_dict['data.filter'] = True )

        label_tensor = torch.tensor(label_val, dtype=torch.int64)
        sample_dict[key_out_gt] = label_tensor  # 'data.ground_truth'

        # add clinical
        features_to_use = self._label_type.select_features()

        clinical_features_to_use = torch.tensor(
            [float(clinical_features[feature]) for feature in features_to_use], dtype=torch.float32
        )
        sample_dict[key_out_clinical_features] = clinical_features_to_use  # 'data.clinical_features'

        if self._is_concat_features_to_input:
            # select input channel
            input_tensor = sample_dict["data.input"]
            input_shape = input_tensor.shape
            for feature in clinical_features_to_use:
                input_tensor = torch.cat(
                    (input_tensor, feature.repeat(input_shape[1], input_shape[2], input_shape[3]).unsqueeze(0)), dim=0
                )
            sample_dict["data.input"] = input_tensor

        return sample_dict


def get_duke_raw_annotations_df(duke_data_dir):
    annotations_path = os.path.join(duke_data_dir, "Annotation_Boxes.csv")
    annotations_df = pd.read_csv(annotations_path)
    return annotations_df


def get_duke_clinical_data_df(duke_data_dir):
    annotations_path = os.path.join(duke_data_dir, "Clinical_and_Other_Features.xlsx")

    df = pd.read_excel(annotations_path, sheet_name="Data", nrows=10)

    columns = []
    col_header = ""
    for i in range(df.shape[1]):
        if not df.columns[i].startswith("Unnamed"):
            col_header = df.columns[i]
        col_name = col_header.strip()

        for row in [0, 1]:
            s = df.iloc[row, i]
            if isinstance(s, str) and len(s.strip()) > 0:
                col_name += ":" + s.strip()
            if col_name not in columns:
                break
        columns.append(col_name)

    annotations_df = pd.read_excel(annotations_path, sheet_name="Data", skiprows=3, header=None, names=columns)
    return annotations_df


def get_samples_for_debug(data_dir, n_pos, n_neg, label_type, sample_ids=None):
    annotations_df = get_duke_clinical_data_df(data_dir).set_index("Patient Information:Patient ID")
    if sample_ids is not None:
        annotations_df = annotations_df.loc[sample_ids]
    label_values = label_type.get_value(annotations_df)
    patient_ids = annotations_df.index
    debug_sample_ids = []
    for label_val, n_vals in zip([True, False], [n_pos, n_neg]):
        debug_sample_ids += patient_ids[label_values == label_val].values.tolist()[:n_vals]
    return debug_sample_ids


def get_series_desc_2_sequence_mapping(metadata_path: str):
    # read metadata file and match between series_desc in metadata file and sequence
    metadata_df = pd.read_csv(metadata_path)
    series_description_list = metadata_df["Series Description"].unique()

    series_desc_2_sequence_mapping = {"ax dyn": "DCE_mix_ph"}

    patterns = ["1st", "2nd", "3rd", "4th"]
    for i_phase in range(1, 5):
        seq_id = f"DCE_mix_ph{i_phase}"
        phase_patterns = [patterns[i_phase - 1], f"{i_phase}ax", f"{i_phase}Ax", f"{i_phase}/ax", f"{i_phase}/Ax"]

        for series_desc in series_description_list:
            has_match = any(p in series_desc for p in phase_patterns)
            if has_match:
                series_desc2 = series_desc.replace(f"{i_phase}ax", f"{i_phase}/ax").replace(
                    f"{i_phase}Ax", f"{i_phase}/Ax"
                )
                series_desc_2_sequence_mapping[series_desc] = seq_id
                series_desc_2_sequence_mapping[series_desc2] = seq_id

    return series_desc_2_sequence_mapping


def get_sample_path(data_path, sample_id):
    sample_path_pattern = os.path.join(data_path, sample_id, "*")
    sample_path = glob.glob(sample_path_pattern)
    assert len(sample_path) == 1
    return sample_path[0]


##################################
class DukeRadiomics(Duke):
    @staticmethod
    def static_pipeline(
        select_series_func, data_dir: Optional[str] = None, verbose: Optional[bool] = True
    ) -> PipelineDefault:
        # remove scaling operator for radiomics calculation
        static_pipline = Duke.static_pipeline(
            data_dir=data_dir,
            select_series_func=select_series_func,
            with_rescale=False,
            output_patch_volumes=False,
            output_stk_volumes=True,
            verbose=verbose,
        )
        return static_pipline

    @staticmethod
    def dynamic_pipeline(
        radiomics_extractor_setting: dict, label_type: Optional[DukeLabelType] = None, verbose: Optional[bool] = False
    ):

        radiomics_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
            **radiomics_extractor_setting
        )  # todo: tal: move to OpExtractRadiomics
        dynamic_steps = [
            (ops_mri.OpDict2Stk(), dict(keys=["data.input.volume4D", "data.input.ref_volume"])),
            (
                ops_mri.OpExtractRadiomics(radiomics_extractor, radiomics_extractor_setting),
                dict(key_in_vol_4d="data.input.volume4D", key_out_radiomics_results="data.radiomics"),
            ),
        ]

        keys_2_keep = ["data.radiomics.original"]
        if label_type is not None:
            key_ground_truth = "data.ground_truth"
            dynamic_steps.append(
                (
                    OpAddDukeLabelAndClinicalFeatures(label_type=label_type),
                    dict(
                        key_in="data.input.annotations",
                        key_out_gt=key_ground_truth,
                        key_out_clinical_features="data.clinical_features",
                    ),
                )
            )
            keys_2_keep.append(key_ground_truth)
        dynamic_steps.append((ops_common.OpKeepKeypaths(), dict(keep_keypaths=keys_2_keep)))

        dynamic_pipeline = PipelineDefault("dynamic", dynamic_steps, verbose=verbose)

        return dynamic_pipeline

    @staticmethod
    def dataset(
        radiomics_extractor_setting: dict,
        label_type: Optional[DukeLabelType] = None,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        select_series_func=get_selected_series_index,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        verbose: Optional[bool] = True,
        cache_kwargs: Optional[dict] = None,
    ) -> DatasetDefault:

        """
        :param label_type: type of label to use
        :param cache_dir: path to store the cache of the static pipeline
        :param data_dir: path to the original data
        :param select_series_func: which series to select for DCE_mix sequences
        :param reset_cache:
        :param num_workers:  number of processes used for caching
        :param sample_ids: list of selected patient_ids for the dataset
        :return:
        """

        if sample_ids is None:
            sample_ids = DukeRadiomics.sample_ids()

        static_pipeline = DukeRadiomics.static_pipeline(
            data_dir=data_dir, select_series_func=select_series_func, verbose=verbose
        )
        dynamic_pipeline = DukeRadiomics.dynamic_pipeline(
            radiomics_extractor_setting=radiomics_extractor_setting, label_type=label_type, verbose=verbose
        )

        if cache_dir is None:
            cacher = None
        else:
            if cache_kwargs is None:
                cache_kwargs = {}
            cacher = SamplesCacher(
                f"duke_cache_ver{Duke.DUKE_DATASET_VER}",
                static_pipeline,
                [cache_dir],
                restart_cache=reset_cache,
                workers=num_workers,
                ignore_nan_inequality=True,
                **cache_kwargs,
            )

        my_dataset = DatasetDefault(
            sample_ids=sample_ids, static_pipeline=static_pipeline, dynamic_pipeline=dynamic_pipeline, cacher=cacher
        )
        my_dataset.create()
        return my_dataset


def remove_entries_with_nan_label(sample_dict, key):
    if np.isnan(sample_dict[key]):
        print(f"====== {get_sample_id(sample_dict)} has nan label ==> excluded")
        return None
    return sample_dict


def get_selected_sample_ids():
    all_sample_ids = Duke.sample_ids()
    excluded_indexes = [
        "029",
        "050",
        "108",
        "122",
        "127",
        "130",
        "138",
        "151",
        "154",
        "155",
        "159",
        "162",
        "171",
        "179",
        "182",
        "194",
        "208",
        "213",
        "222",
        "226",
        "243",
        "248",
        "257",
        "272",
        "276",
        "279",
        "302",
        "309",
        "314",
        "332",
        "347",
        "359",
        "367",
        "382",
        "388",
        "391",
        "406",
        "422",
        "434",
        "447",
        "449",
        "470",
        "524",
        "549",
        "553",
        "555",
        "571",
        "579",
        "600",
        "619",
        "621",
        "627",
        "637",
        "638",
        "658",
        "701",
        "719",
        "733",
        "747",
        "775",
        "779",
        "785",
        "810",
        "813",
        "828",
        "837",
        "848",
        "867",
        "918",
        "919",
    ]
    excluded_sample_ids = set(["Breast_MRI_" + s for s in excluded_indexes])

    selected_samples_id = [s for s in all_sample_ids if s not in excluded_sample_ids]
    return selected_samples_id
