import numpy as np
import glob
import os
import logging

import pickle
from typing import Hashable, Optional, Sequence

import pandas as pd
from functools import partial

from fuse.data import DatasetDefault
from fuse.data import PipelineDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.op_base import OpReversibleBase, OpBase
from fuse.data.ops import ops_read, ops_cast
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuseimg.data.ops import ops_mri
from fuse.data.ops import ops_common
from fuse.data.ops.ops_aug_common import OpSample, OpRandApply
from fuse.utils.rand.param_sampler import RandBool, RandInt, Uniform, Choice

from fuseimg.data.ops.aug import geometry, geometry3d

from enum import Enum

import torch


def get_selected_series_index(sample_id, seq_id):
    patient_id = sample_id[:-2]
    map = {"T2": -1, "ADC": 0, "ktrans": 0, "MASK": 0}
    if patient_id in ["ProstateX-0148", "ProstateX-0180"]:
        map["b"] = [1, 2]
    elif patient_id in ["ProstateX-0191"]:
        map["b"] = [0, 0]
    elif patient_id in ["ProstateX-0116"]:
        map["b"] = [0, 1]
    else:
        map["b"] = [0, 2]
    return map[seq_id]


class ProstateXLabelType(Enum):
    ClinSig = "ClinSig"

    def get_features_to_use(self):
        if self == ProstateXLabelType.ClinSig:
            return ["zone"]

        raise NotImplementedError(self)

    def get_column_name(self):
        if self == ProstateXLabelType.ClinSig:
            return "ClinSig"

    def get_process_func(self):
        if self == ProstateXLabelType.ClinSig:
            return lambda val: 1 if val > 0 else 0

        raise NotImplementedError(self)

    def get_value(self, clinical_features):
        col_name = self.get_column_name()
        value = clinical_features[col_name] + 0  # why should be add 0??
        process_func = self.get_process_func()
        if process_func is None:
            return value
        if isinstance(value, pd.Series):
            value = value.apply(process_func)
        else:
            value = process_func(value)
        return value

    def get_num_classes(self):
        return 2  # currrently all are binary classification tasks


class ProstateX:
    ProstateX_DATASET_VER = 0
    PATCH_XY_SIZE = 74
    PATCH_Z_SIZE = 13

    @staticmethod
    def sample_ids(data_dir):
        annotations_df = get_prostate_x_annotations_df(data_dir)
        return annotations_df["Sample ID"].values

    @staticmethod
    def static_pipeline(
        root_path,
        select_series_func,
        with_rescale: Optional[bool] = True,
        keep_stk_volumes: Optional[bool] = False,
        verbose: Optional[bool] = True,
        annotations_df: Optional[pd.DataFrame] = None,
    ) -> PipelineDefault:

        data_path = os.path.join(root_path, "PROSTATEx")
        if annotations_df is None:
            annotations_df = get_prostate_x_annotations_df(root_path)

        series_desc_2_sequence_map = get_series_desc_2_sequence_mapping()
        dicom_seq_ids = ["T2", "b", "b_mix", "ADC"]
        seq_reverse_map = {s: True for s in dicom_seq_ids}

        static_pipeline_steps = [
            # step 1: map sample_ids to
            (
                OpProstateXSampleIDDecode(data_path=data_path),
                dict(key_path_out="data.input.mri_path", key_patient_id_out="data.input.patient_id"),
            ),
            # step 2: read files info for the sequences
            (
                ops_mri.OpExtractDicomsPerSeq(
                    seq_ids=dicom_seq_ids,
                    series_desc_2_sequence_map=series_desc_2_sequence_map,
                    use_order_indicator=False,
                ),
                dict(
                    key_in="data.input.mri_path",
                    key_out_seq_ids="data.input.seq_ids",
                    key_out_sequence_prefix="data.input.sequence.",
                ),
            ),
            # step 3: Load STK volumes of MRI sequences
            (
                ops_mri.OpLoadDicomAsStkVol(seq_reverse_map=seq_reverse_map),
                dict(key_in_seq_ids="data.input.seq_ids", key_sequence_prefix="data.input.sequence."),
            ),
            # step 4: rename sequence b_mix (if exists) to b; fix certain b sequences
            (
                ops_common.OpLambda(
                    func=partial(
                        ops_mri.rename_seqeunce_from_dict,
                        seq_id_old="b_mix",
                        seq_id_new="b",
                        key_sequence_prefix="data.input.sequence.",
                        key_seq_ids="data.input.seq_ids",
                    )
                ),
                dict(key=None),
            ),
            # step 5: read ktrans
            (
                ops_mri.OpReadSTKImage(
                    seq_id="ktrans", get_image_file=partial(get_ktrans_image_file_from_sample_id, data_dir=root_path)
                ),
                dict(key_sequence_prefix="data.input.sequence.", key_seq_ids="data.input.seq_ids"),
            ),
            # step 6: select single volume from b_mix/T2 sequence
            (
                ops_mri.OpSelectVolumes(
                    get_indexes_func=select_series_func,
                    delete_input_volumes=True,
                    selected_seq_ids=["T2", "b", "ADC", "ktrans"],
                ),
                dict(
                    key_in_seq_ids="data.input.seq_ids",
                    key_in_sequence_prefix="data.input.sequence.",
                    key_out_volumes="data.input.selected_volumes",
                    key_out_volumes_info="data.input.selected_volumes_info",
                ),
            ),
            # step 7
            (
                ops_common.OpLambda(
                    func=partial(
                        fix_certain_b_sequences,
                        key_in_volumes_info="data.input.selected_volumes_info",
                        key_volumes="data.input.selected_volumes",
                    )
                ),
                dict(key=None),
            ),
            # step 8: set reference volume to be first and register other volumes with respect to it
            (
                ops_mri.OpResampleStkVolsBasedRef(reference_inx=0, interpolation="bspline"),
                dict(key="data.input.selected_volumes"),
            ),
            # step 9: create a single 4D volume from all the sequences (4th channel is the sequence)
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
            # step 10:
            static_pipeline_steps += [(ops_mri.OpRescale4DStk(), dict(key="data.input.volume4D"))]

        static_pipeline_steps += [
            # step 11: read tabular data for each patch
            (
                ops_read.OpReadDataframe(data=annotations_df, key_column="Sample ID"),
                dict(key_out_group="data.input.patch_annotations"),
            ),
            # step 12: create patch volumes: (i) fixed size around center of annotatins (orig), and (ii) entire annotations
            (
                ops_mri.OpCreatePatchVolumes(
                    lsn_shape=(ProstateX.PATCH_Z_SIZE, ProstateX.PATCH_XY_SIZE, ProstateX.PATCH_XY_SIZE),
                    name_suffix="_T0",
                    pos_key="pos",
                    lsn_spacing=(3, 0.5, 0.5),
                    crop_based_annotation=False,
                    delete_input_volumes=not keep_stk_volumes,
                ),
                dict(
                    key_in_volume4D="data.input.volume4D",
                    key_in_ref_volume="data.input.ref_volume",
                    key_in_patch_annotations="data.input.patch_annotations",
                    key_out="data.input.patch_volume",
                ),
            ),
        ]
        if keep_stk_volumes:
            static_pipeline_steps += [
                # step 13: move to ndarray - to allow quick saving
                (
                    ops_mri.OpStk2Dict(),
                    dict(keys=["data.input.patient_id", "data.input.volume4D", "data.input.ref_volume"]),
                )
            ]
        static_pipeline = PipelineDefault("static", static_pipeline_steps, verbose=verbose)

        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        train: bool,
        label_type: Optional[ProstateXLabelType] = None,
        num_channels: int = 5,
        verbose: Optional[bool] = True,
    ):
        volume_key = "data.input.patch_volume"
        dynamic_steps = [(ops_cast.OpToTensor(), dict(key=volume_key, dtype=torch.float32))]

        keys_2_keep = [volume_key, "data.input.patient_id"]
        if label_type is not None:
            key_ground_truth = "data.ground_truth"
            dynamic_steps.append(
                (
                    OpAdProstateXLabelAndClinicalFeatures(label_type=label_type),
                    dict(
                        key_in="data.input.patch_annotations",
                        key_out_gt=key_ground_truth,
                        key_out_clinical_features="data.clinical_features",
                    ),
                )
            )
            keys_2_keep.append(key_ground_truth)

        dynamic_steps.append((ops_common.OpKeepKeypaths(), dict(keep_keypaths=keys_2_keep)))

        # augmentation

        if train:
            dynamic_steps += [
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
                # [
                #     ('data.input',),
                #     squeeze_3d_to_2d,
                #     {'axis_squeeze': 'z'},
                #     {}
                # ],
                (geometry.OpAugSqueeze3Dto2D(), dict(key="data.input.patch_volume", axis_squeeze=1)),
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
                # [
                #     ('data.input',),
                #     aug_op_affine,
                #     {'rotate': Uniform(-3.0, 3.0),
                #      'translate': (RandInt(-2, 2), RandInt(-2, 2)),
                #      'flip': (False, False),
                #      'scale': Uniform(0.9, 1.1),
                #      'channels': Choice(image_channels, probabilities=None)},
                #     {'apply': RandBool(0.5) if train_common_params['data.aug.phase_misalignment'] else 0}
                # ],
                (
                    OpRandApply(OpSample(geometry.OpAugAffine2D()), 0.5),
                    dict(
                        key="data.input.patch_volume",
                        rotate=Uniform(-3.0, 3.0),
                        scale=Uniform(0.9, 1.1),
                        flip=(False, False),
                        translate=(RandInt(-2, 2), RandInt(-2, 2)),
                        channels=Choice(
                            [list(range(0, ProstateX.PATCH_Z_SIZE))]
                        ),  # todo: but we are 2D - there are no channels??
                    ),
                ),
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

        dynamic_pipeline = PipelineDefault("dynamic", dynamic_steps, verbose=verbose)

        return dynamic_pipeline

    @staticmethod
    def dataset(
        label_type: Optional[ProstateXLabelType] = None,
        train: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        select_series_func=get_selected_series_index,
        num_channels: int = 5,
        reset_cache: bool = False,
        num_workers: int = 10,
        sample_ids: Optional[Sequence[Hashable]] = None,
        verbose: Optional[bool] = True,
        annotations_df: Optional[pd.DataFrame] = None,
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

        if data_dir is None:
            data_dir = "/projects/msieve/MedicalSieve/PatientData/ProstateX/manifest-A3Y4AE4o5818678569166032044/"

        if sample_ids is None:
            sample_ids = ProstateX.sample_ids(data_dir)

        static_pipeline = ProstateX.static_pipeline(
            root_path=data_dir, select_series_func=select_series_func, annotations_df=annotations_df, verbose=verbose
        )
        dynamic_pipeline = ProstateX.dynamic_pipeline(
            train=train, label_type=label_type, num_channels=num_channels, verbose=verbose
        )

        if cache_dir is None:
            cacher = None
        else:
            if cache_kwargs is None:
                cache_kwargs = {}
            cacher = SamplesCacher(
                f"prostate_x_cache_ver{ProstateX.ProstateX_DATASET_VER}",
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


def fix_certain_b_sequences(sample_dict, key_in_volumes_info, key_volumes):
    volumes_info = sample_dict[key_in_volumes_info]
    volumes = sample_dict[key_volumes]

    B_SER_FIX = [
        "diffusie-3Scan-4bval_fs",
        "ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen",
        "diff tra b 50 500 800 WIP511b alle spoelen",
    ]
    # seq = ['T2', 'b400', 'b800', 'ADC', 'ktrans']
    # if ('b' in seq_info.keys()):
    #     if (seq_info['b'][0] in B_SER_FIX):
    #         vols_list[seq.index('b800')].CopyInformation(vols_list[seq.index('ADC')])
    #         vols_list[seq.index('b400')].CopyInformation(vols_list[seq.index('ADC')])

    has_b_to_fix = np.any([(info["seq_id"] == "b") and (info["series_desc"] in B_SER_FIX) for info in volumes_info])
    if has_b_to_fix:
        for i, val in zip([1, 2], [400, 800]):
            volumes[i].CopyInformation(volumes[3])
            assert volumes_info[i]["seq_id"] == "b"
            assert volumes_info[i]["series_desc"] in B_SER_FIX
            val2 = volumes_info[i]["dicoms_id"]
            if val == val2:
                print(f"fix B OK 8888888888  : {val}={val2}")
            else:
                print(f"fix B mismatch 8888888888  : {val}!={val2}")

    return sample_dict


def get_ktrans_image_file_from_sample_id(sample_id, data_dir):
    patient_id = sample_id.split("_")[0]
    ktrans_patient_dir = os.path.join(data_dir, "ProstateXKtrains-train-fixed", patient_id)
    ktrans_mhd_files = glob.glob(os.path.join(ktrans_patient_dir, "*.mhd"))
    assert len(ktrans_mhd_files) == 1
    return ktrans_mhd_files[0]


class OpProstateXSampleIDDecode(OpReversibleBase):
    """
    decodes sample id into path of MRI images
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path

    def __call__(self, sample_dict: NDict, key_path_out: str, key_patient_id_out: str, op_id: Optional[str]) -> NDict:
        sid = get_sample_id(sample_dict)

        patient_id = sid[:-2]
        sample_dict[key_patient_id_out] = get_sample_path(self._data_path, patient_id)
        sample_dict[key_path_out] = get_sample_path(self._data_path, patient_id)

        return sample_dict

    def reverse(self, sample_dict: dict, key_to_reverse: str, key_to_follow: str, op_id: Optional[str]) -> dict:
        return sample_dict


class OpAdProstateXLabelAndClinicalFeatures(OpBase):
    """
    decodes sample id into path of MRI images
    """

    def __init__(self, label_type: ProstateXLabelType, is_concat_features_to_input: Optional[bool] = False, **kwargs):
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

        if False:
            # add clinical
            features_to_use = self._label_type.get_features_to_use()

            zone2feature = {
                "PZ": torch.tensor(np.array([0, 0, 0]), dtype=torch.float32),
                "TZ": torch.tensor(np.array([0, 0, 1]), dtype=torch.float32),
                "AS": torch.tensor(np.array([0, 1, 0]), dtype=torch.float32),
                "SV": torch.tensor(np.array([1, 0, 0]), dtype=torch.float32),
            }

            clinical_features_to_use = zone2feature[clinical_features[features_to_use[0]]]

            sample_dict[key_out_clinical_features] = clinical_features_to_use  # 'data.clinical_features'

            if self._is_concat_features_to_input:
                # select input channel
                input_tensor = sample_dict["data.input"]
                input_shape = input_tensor.shape
                for feature in clinical_features_to_use:
                    input_tensor = torch.cat(
                        (input_tensor, feature.repeat(input_shape[1], input_shape[2], input_shape[3]).unsqueeze(0)),
                        dim=0,
                    )
                sample_dict["data.input"] = input_tensor

        return sample_dict


def get_prostate_x_annotations_df(data_dir):
    if True:
        # v5
        annotations_df = pd.read_csv(os.path.join(data_dir, "Lesion Information", "ProstateX-Findings-Train.csv"))
        annotations_df["Patient ID"] = annotations_df["ProxID"]
        annotations_df = annotations_df.set_index("ProxID")
        annotations_df = annotations_df[["Patient ID", "fid", "ClinSig", "pos", "zone"]]
        # ['ProstateX-0005_1',
        # 'ProstateX-0025_1', 'ProstateX-0025_2', 'ProstateX-0025_3', 'ProstateX-0025_4',
        # 'ProstateX-0105_2', 'ProstateX-0105_3', 'ProstateX-0154_3']

        vals = [
            ("ProstateX-0005", 1),  # there are two of this
            ("ProstateX-0105", 2),
            ("ProstateX-0105", 3),
            ("ProstateX-0154", 3),
        ]
        a_filter = np.zeros(annotations_df.shape[0], dtype=bool)
        for pid, fid in vals:
            a_filter |= (annotations_df["Patient ID"] == pid) & (annotations_df["fid"] == fid)
        assert a_filter.sum() == 5
        annotations_df = annotations_df[~a_filter]

    else:
        # v6
        PROSTATEX_PROCESSED_FILE_DIR = "/projects/msieve_dev3/usr/Tal/prostate_x_processed_files"

        annotations_path = os.path.join(
            PROSTATEX_PROCESSED_FILE_DIR, "dataset_prostate_x_folds_ver29062021_seed1.pickle"
        )
        with open(annotations_path, "rb") as infile:
            fold_annotations_dict = pickle.load(infile)
        annotations_df = pd.concat(
            [fold_annotations_dict[f"data_fold{fold}"] for fold in range(len(fold_annotations_dict))]
        )
        annotations_df = annotations_df[["Patient ID", "fid", "ClinSig", "pos", "zone"]]

    pids_to_fix = [("ProstateX-0159", 3)]

    # fix a bug in the dataset of wrong indexing
    for pid, n_fid in pids_to_fix:
        a_filter = annotations_df["Patient ID"] == pid
        assert a_filter.sum() == n_fid
        annotations_df.loc[a_filter, "fid"] = np.arange(1, n_fid + 1)
    annotations_df["Sample ID"] = annotations_df["Patient ID"] + "_" + annotations_df["fid"].astype(str)

    # filter problematic samples:
    problematic_patient_ids = ["ProstateX-0025"]
    a_filter = ~annotations_df["Patient ID"].isin(problematic_patient_ids)
    if not np.all(a_filter):
        annotations_df = annotations_df[a_filter]
    return annotations_df


def get_samples_for_debug(n_pos, n_neg, label_type):
    annotations_df = get_prostate_x_annotations_df()
    label_values = label_type.get_value(annotations_df)
    patient_ids = annotations_df["Sample ID"]
    sample_ids = []
    for label_val, n_vals in zip([True, False], [n_pos, n_neg]):
        sample_ids += patient_ids[label_values == label_val].values.tolist()[:n_vals]
    return sample_ids


def get_series_desc_2_sequence_mapping():
    series_desc_2_sequence_mapping = {
        "t2_tse_tra": "T2",
        "t2_tse_tra_Grappa3": "T2",
        "t2_tse_tra_320_p2": "T2",
        "ep2d-advdiff-3Scan-high bvalue 100": "b",
        "ep2d-advdiff-3Scan-high bvalue 500": "b",
        "ep2d-advdiff-3Scan-high bvalue 1400": "b",
        "ep2d_diff_tra2x2_Noise0_FS_DYNDISTCALC_BVAL": "b",
        "ep2d_diff_tra_DYNDIST": "b_mix",
        "ep2d_diff_tra_DYNDIST_MIX": "b_mix",
        "diffusie-3Scan-4bval_fs": "b_mix",
        "ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen": "b_mix",
        "diff tra b 50 500 800 WIP511b alle spoelen": "b_mix",
        "ep2d_diff_tra_DYNDIST_MIX_ADC": "ADC",
        "diffusie-3Scan-4bval_fs_ADC": "ADC",
        "ep2d-advdiff-MDDW-12dir_spair_511b_ADC": "ADC",
        "ep2d-advdiff-3Scan-4bval_spair_511b_ADC": "ADC",
        "ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen_ADC": "ADC",
        "diff tra b 50 500 800 WIP511b alle spoelen_ADC": "ADC",
        "ADC_S3_1": "ADC",
        "ep2d_diff_tra_DYNDIST_ADC": "ADC",
    }

    return series_desc_2_sequence_mapping


def get_sample_path(data_path, patient_id):
    patient_path_pattern = os.path.join(data_path, patient_id, "*")
    patient_path = sorted(glob.glob(patient_path_pattern))
    if len(patient_path) > 1:
        lgr = logging.getLogger("Fuse")
        lgr.warning(f"{patient_id} has {len(patient_path)} files. Taking first")
    return patient_path[0]


##################################
