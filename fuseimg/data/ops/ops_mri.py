import glob
import os
from typing import Optional

import SimpleITK as sitk
import h5py

import numpy as np
import pydicom

from fuse.data import OpBase, get_sample_id
from fuse.utils import NDict
from typing import Tuple
import torch
import cv2



class OpExtractDicomsPerSeq(OpBase):

    def __init__(self, seq_ids, series_desc_2_sequence_map, use_order_indicator: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._seq_ids = seq_ids
        self._series_desc_2_sequence_map = series_desc_2_sequence_map
        self._use_order_indicator = use_order_indicator



    def __call__(self, sample_dict: NDict, key_in:str,  key_out_sequences:str,
                 key_out_path_prefix:str, key_out_dicoms_prefix: str, key_out_series_num_prefix:str):
        sample_path = sample_dict[key_in]

        seq_2_info_map = extract_seq_2_info_map(sample_path, self._series_desc_2_sequence_map)
        for seq_id in self._seq_ids:
            if key_out_sequences not in sample_dict:
                sample_dict[key_out_sequences] = []
            sample_dict[key_out_sequences].append(seq_id)

            seq_info_list = seq_2_info_map.get(seq_id, None)
            if seq_info_list is None:
                # sequence does not exist for the patient
                continue

            sample_dict[f'{key_out_path_prefix}{seq_id}'] = []
            sample_dict[f'{key_out_dicoms_prefix}{seq_id}'] = []
            sample_dict[f'{key_out_series_num_prefix}{seq_id}'] = []
            for seq_info in seq_info_list:

                sorted_dicom_list = sort_dicoms_by_field(seq_info['path'],  seq_info['dicom_field'], self._use_order_indicator)

                sample_dict[f'{key_out_path_prefix}{seq_id}'].append(seq_info['path'])
                sample_dict[f'{key_out_dicoms_prefix}{seq_id}'] += sorted_dicom_list
                sample_dict[f'{key_out_series_num_prefix}{seq_id}'].append(seq_info['series_num'])


        return sample_dict



######################################################################
class OpLoadDicomAsStkVol(OpBase):
    '''
    Return location dir of requested sequence
    '''

    def __init__(self,reverse_order: bool=False, is_file: bool=False, **kwargs):
        """
        :param reverse_order: sometimes reverse dicoms orders is needed
        (for b series in which more than one sequence is provided inside the img_path)
        :param is_file: if True loads all dicoms from img_path
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._reverse_order = reverse_order
        self._is_file = is_file

    def __call__(self, sample_dict: NDict, key_in_seq_ids:str, key_in_path_prefix: str, key_in_dicoms_prefix: str,
                 key_out_prefix: str, reverse_order: Optional[bool]=None, is_file:Optional[bool]=None):
        """
        extract_stk_vol loads dicoms into sitk vol
        :param img_path: path to dicoms - load all dicoms from this path
        :param img_list: list of dicoms to load
        :return: list of stk vols
        """
        if is_file is None:
            is_file = self._is_file
        if reverse_order is None:
            reverse_order = self._reverse_order

        seq_ids = sample_dict[key_in_seq_ids]

        for seq_id in seq_ids:

            img_paths = sample_dict.get(f'{key_in_path_prefix}{seq_id}')
            if img_paths is None:
                continue

            dicom_files_list = sample_dict.get(f'{key_in_dicoms_prefix}{seq_id}')
            assert len(dicom_files_list) ==  len(img_paths)
            stk_vols = []
            for i, img_path in enumerate(img_paths):
                dicom_files = dicom_files_list[i]
                stk_vol = get_stk_volume(img_path, is_file, dicom_files, reverse_order)
                stk_vols.append(stk_vol)
            sample_dict[f'{key_out_prefix}{seq_id}'] = stk_vols

        return sample_dict


def get_stk_volume(img_path, is_file, dicom_files, reverse_order):

    # load from HDF5
    if img_path[-4::] in 'hdf5':
        vol = _read_HDF5_file(img_path)
        return vol

    if is_file:
        vol = sitk.ReadImage(img_path)
        return vol


    series_reader = sitk.ImageSeriesReader()

    if dicom_files is None:
        dicom_files = series_reader.GetGDCMSeriesFileNames(img_path)


    if isinstance(dicom_files, str):
        dicom_files = [dicom_files]
    if img_path not in dicom_files[0]:
        dicom_files = [os.path.join(img_path, dicom_file) for dicom_file in dicom_files]
    dicom_files = dicom_files[::-1] if reverse_order else dicom_files
    series_reader.SetFileNames(dicom_files)
    vol = series_reader.Execute()
    return vol



def _read_HDF5_file(img_path):
    with h5py.File(img_path, 'r') as hf:
        _array = np.array(hf['array'])
        _spacing = hf.attrs['spacing']
        _origin = hf.attrs['origin']
        _world_matrix = np.array(hf.attrs['world_matrix'])[:3, :3]
        _world_matrix_unit = _world_matrix / np.linalg.norm(_world_matrix, axis=0)
        _world_matrix_unit_flat = _world_matrix_unit.flatten()

    # volume 2 sitk
    vol = sitk.GetImageFromArray(_array)
    vol.SetOrigin([_origin[i] for i in [1, 2, 0]])
    vol.SetDirection(_world_matrix_unit_flat)
    vol.SetSpacing([_spacing[i] for i in [1, 2, 0]])
    return vol


#############################
class OpGroupDCESequences(OpBase):
    def __init__(self, verbose: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._verbose = verbose

    def __call__(self, sample_dict: NDict, key_sequence_ids: str,
                 key_volumes_prefix: str, key_path_prefix: str, key_series_num_prefix: str):

        """
        extract_list_of_rel_vol extract the volume per seq based on SER_INX_TO_USE
        and put in one list
        :param vols_dict: dict of sitk vols per seq
        :param seq_info: dict of seq description per seq
        :return:

        """

        seq_ids = sample_dict[key_sequence_ids]

        # delete_seqeunce_from_dict(seq_id='b_mix', sample_dict=sample_dict, seq_ids=seq_ids) #todo: no need to delete if not included. Need to be specified by the user

        all_dce_mix_ph_sequences = [f'DCE_mix_ph{i}' for i in range(1, 5)] + ['DCE_mix_ph']
        # seq_id_exist_func = lambda seq_id: seq_id in seq_ids and len(
        #     sample_dict[f'{key_in_volumes_prefix}{seq_id}']) > 0

        existing_dce_mix_ph_sequences = [seq_id for seq_id in all_dce_mix_ph_sequences if f'{key_path_prefix}{seq_id}' in sample_dict]
        # handle multiphase DCE in different series
        if existing_dce_mix_ph_sequences:
            new_seq_id = 'DCE_mix'
            key_path = f'{key_path_prefix}{new_seq_id}'
            key_volumes = f'{key_volumes_prefix}{new_seq_id}'
            key_series_num = f'{key_series_num_prefix}{new_seq_id}'

            assert new_seq_id in seq_ids
            assert key_path not in sample_dict

            sample_dict[key_path] = []
            sample_dict[key_volumes] = []
            sample_dict[key_series_num] = []


            for seq_id in existing_dce_mix_ph_sequences:
                seq_paths = _get_as_list(sample_dict[f'{key_path_prefix}{seq_id}'])
                stk_vols = _get_as_list(sample_dict[f'{key_volumes_prefix}{seq_id}'])
                series_num = _get_as_list(sample_dict[f'{key_series_num_prefix}{seq_id}'])

                if seq_id == 'DCE_mix_ph':
                    inx_sorted = np.argsort(series_num)
                    for ser_num_inx in inx_sorted:
                        sample_dict[key_volumes].append(stk_vols[ser_num_inx])
                        sample_dict[key_path].append(seq_paths[ser_num_inx])
                        sample_dict[key_series_num].append(series_num[ser_num_inx])
                else:
                    sample_dict[key_path] += seq_paths
                    sample_dict[key_volumes] += stk_vols
                    sample_dict[key_series_num] += series_num

        for seq_id in all_dce_mix_ph_sequences:
            _delete_seqeunce_from_dict(seq_id=seq_id, sample_dict=sample_dict, key_sequence_ids=key_sequence_ids)

        return sample_dict

#############################
class OpSelectVolumes(OpBase):
    def __init__(self, get_indexes_func, delete_input_volumes: Optional[bool]=False, verbose: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._get_indexes_func = get_indexes_func
        self._delete_input_volumes = delete_input_volumes
        self._verbose = verbose

    def __call__(self, sample_dict: NDict,key_in_sequence_ids:str,
                 key_in_volumes_prefix: str, key_in_path_prefix: str,
                 key_out_volumes: str, key_out_paths: str):


        sample_id =get_sample_id(sample_dict)
        seq_ids = sample_dict[key_in_sequence_ids]

        sample_dict[f'{key_out_volumes}'] = []
        sample_dict[f'{key_out_paths}'] = []
        for seq_id in seq_ids:
            if f'{key_in_volumes_prefix}{seq_id}' in sample_dict:
                key_in_volumes = f'{key_in_volumes_prefix}{seq_id}'

                seq_volumes = _get_as_list(sample_dict[key_in_volumes])
                seq_path = _get_as_list(sample_dict[f'{key_in_path_prefix}{seq_id}'])

                if self._delete_input_volumes:
                    del sample_dict[key_in_volumes]
            else:
                seq_volumes = []
                seq_path = []
            vol_inx_to_use = _get_as_list(self._get_indexes_func(sample_id, seq_id))

            for inx in vol_inx_to_use:
                if len(seq_volumes)  == 0:
                    assert len(sample_dict[key_out_volumes]) > 0
                    seq_volume_template = sample_dict[key_out_volumes][0]
                    volume = get_zeros_vol(seq_volume_template)
                    a_path = 'NAN'
                else:
                    if inx > len(seq_volumes) - 1:
                        inx = -1  # take the last
                    a_path = seq_path[inx]

                    volume = seq_volumes[inx]
                    if len(volume) == 0:
                        volume = get_zeros_vol(seq_volumes[0])
                        if self._verbose:
                            print(f'\n - problem with reading {seq_id} volume!')

                sample_dict[key_out_volumes].append(volume)
                sample_dict[key_out_paths].append(a_path)


        return sample_dict




############################

class OpResampleStkVolsBasedRef(OpBase):
    def __init__(self, reference_inx: int, interpolation: str, **kwargs):
        super().__init__(**kwargs)
        assert reference_inx is not None  # todo: redundant??
        self.reference_inx = reference_inx
        self.interpolation = interpolation

    def __call__(self, sample_dict: NDict,
                 key: str):

        # ------------------------
        # create resampling operator based on ref vol
        volumes = sample_dict[key]
        assert len(volumes) >0
        # if self.reference_inx > 0:
        #     volumes = [volumes[self.reference_inx]]+ volumes[:self.reference_inx]+ volumes[volumes+1:]

        seq_volumes_resampled = [ sitk.Cast(v, sitk.sitkFloat32) for v in volumes]
        ref_volume = volumes[self.reference_inx]

        resample = self.create_resample(ref_volume, self.interpolation, size=ref_volume.GetSize(),
                                        spacing=ref_volume.GetSpacing())

        for i in range(len(seq_volumes_resampled)):
            if i == self.reference_inx:
                continue

            seq_volumes_resampled[i] = resample.Execute(seq_volumes_resampled[i])

        sample_dict[key] = seq_volumes_resampled
        return sample_dict

    def create_resample(self, vol_ref: sitk.sitkFloat32, interpolation: str, size: Tuple[int, int, int],
                        spacing: Tuple[float, float, float]):
        """
        create_resample create resample operator
        :param vol_ref: sitk vol to use as a ref
        :param interpolation:['linear','nn','bspline']
        :param size: in pixels ()
        :param spacing: in mm ()
        :return: resample sitk operator
        """

        if interpolation == 'linear':
            interpolator = sitk.sitkLinear
        elif interpolation == 'nn':
            interpolator = sitk.sitkNearestNeighbor
        elif interpolation == 'bspline':
            interpolator = sitk.sitkBSpline

        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(vol_ref)
        resample.SetOutputSpacing(spacing)
        resample.SetInterpolator(interpolator)
        resample.SetSize(size)
        return resample
#######

class OpStackList4DStk(OpBase):
    def __init__(self, delete_input_volumes: Optional[bool]=False, reference_inx: Optional[int]=0, **kwargs):
        super().__init__(**kwargs)
        self._reference_inx = reference_inx
        self._delete_input_volumes = delete_input_volumes

    def __call__(self, sample_dict: NDict, key_in:str, key_out_volume4d:str, key_out_ref_volume: str):
        vols_stk_list = sample_dict[key_in]
        if self._delete_input_volumes:
            del sample_dict[key_in]

        vol_arr = [sitk.GetArrayFromImage(vol) for vol in vols_stk_list]
        vol_final = np.stack(vol_arr, axis=-1)
        vol_final_sitk = sitk.GetImageFromArray(vol_final, isVector=True)
        vol_final_sitk.CopyInformation(vols_stk_list[self._reference_inx])

        sample_dict[key_out_volume4d] = vol_final_sitk
        sample_dict[key_out_ref_volume] = vols_stk_list[self._reference_inx]
        return sample_dict


class OpRescale4DStk(OpBase):
    def __init__(self, mask_ch_inx: Optional[int]=-1, thres: Optional[tuple]=(1.0, 99.0), method: Optional[str]='noclip', **kwargs):
        super().__init__(**kwargs)
        self._mask_ch_inx = mask_ch_inx
        self._thres = thres
        self._method=method

    def __call__(self, sample_dict: NDict, key: str):
        stk_vol_4D = sample_dict[key]

        vol_backup = sitk.Image(stk_vol_4D)
        vol_array = sitk.GetArrayFromImage(stk_vol_4D)
        if len(vol_array.shape) < 4:
            vol_array = vol_array[:, :, :, np.newaxis]
        vol_array_pre_rescale = vol_array.copy()
        vol_array = apply_rescaling(vol_array, thres=self._thres, method=self._method)


        if self._mask_ch_inx:
            bool_mask = np.zeros(vol_array_pre_rescale[:, :, :, self._mask_ch_inx].shape)
            bool_mask[vol_array_pre_rescale[:, :, :, self._mask_ch_inx] > 0.3] = 1
            vol_array[:, :, :, self._mask_ch_inx] = bool_mask

        vol_final = sitk.GetImageFromArray(vol_array, isVector=True)
        vol_final.CopyInformation(vol_backup)
        vol_final = sitk.Image(vol_final)
        sample_dict[key] = vol_final
        return sample_dict



class OpAddPatchesData(OpBase):
    def __init__(self, get_annotations_func, **kwargs):
        super().__init__(**kwargs)
        self._get_annotations_func = get_annotations_func


    def __call__(self, sample_dict: NDict, key_out: str):
        sample_id  = get_sample_id(sample_dict)
        annotations_df = self._get_annotations_func(sample_id)
        sample_dict[key_out] = [annotations_df.iloc[i] for i in range(annotations_df.shape[0])]

        return sample_dict



class OpCreatePatcheVolumes(OpBase):
    def __init__(self, lsn_shape, lsn_spacing, longtd_inx: Optional[int] = 0, **kwargs):
        super().__init__(**kwargs)
        self._lsn_shape = lsn_shape
        self._lsn_spacing = lsn_spacing
        self._longtd_inx = longtd_inx

    def __call__(self, sample_dict: NDict, key_in_volume4D: str, key_in_ref_volume: str, key_in_patch_rows: str,
                 key_out_cropped_vol_by_mask: str, key_out_cropped_vol: str):

        vol_ref = sample_dict[key_in_ref_volume]
        vol_4D = sample_dict[key_in_volume4D]
        patch_row_list = sample_dict[key_in_patch_rows]

        sample_dict[key_out_cropped_vol_by_mask] = []
        sample_dict[key_out_cropped_vol] = []

        for row in patch_row_list:
            # read original position
            pos_orig = np.fromstring(row[f'centroid_T{self._longtd_inx}'][1:-1], dtype=np.float32, sep=',')

            # transform to pixel coordinate in ref coords
            pos_vol = np.array(vol_ref.TransformPhysicalPointToContinuousIndex(pos_orig.astype(np.float64)))

            vol_4d_tmp = sitk.GetArrayFromImage(vol_4D)
            if sum(sum(sum(vol_4d_tmp[:, :, :, -1]))) == 0:  # if the mast does not exist
                bbox_coords = np.fromstring(row[f'bbox_T{self._longtd_inx}'][1:-1], dtype=np.int32, sep=',')
                mask = extract_mask_from_annotation(vol_ref, bbox_coords)
                vol_4d_tmp[:, :, :, -1] = mask
                vol_4d_new = sitk.GetImageFromArray(vol_4d_tmp)
                vol_4D = vol_4d_new

            for is_use_mask in [False, True]:
                if is_use_mask:
                    cropped_vol_size=(self._lsn_shape[2], self._lsn_shape[1], self._lsn_shape[0])
                else:
                    cropped_vol_size = (2 * self._lsn_shape[2], 2 * self._lsn_shape[1], self._lsn_shape[0])

                vol_cropped = crop_lesion_vol_mask_based(vol_4D, pos_vol, vol_ref,
                                        size=cropped_vol_size,
                                        spacing=(self._lsn_spacing[2], self._lsn_spacing[1], self._lsn_spacing[0]),
                                        mask_inx=-1,
                                        is_use_mask=is_use_mask)

                vol_cropped_arr = sitk.GetArrayFromImage(vol_cropped)
                if len(vol_cropped_arr.shape) < 4:
                    # fix dimensions in case of one seq
                    vol_cropped_arr = vol_cropped_arr[:, :, :, np.newaxis]
                    vol_cropped_arr = np.moveaxis(vol_cropped_arr, 3, 0)
                else:
                    vol_cropped_arr = np.moveaxis(vol_cropped_arr, 3, 0)

                if np.isnan(vol_cropped_arr).any():
                    input[np.isnan(input)] = 0


                key_out = key_out_cropped_vol_by_mask if is_use_mask else key_out_cropped_vol
                sample_dict[key_out].append(vol_cropped_arr)


        return sample_dict





#######


class OpStk2Torch(OpBase):

    def __call__(self, sample_dict: NDict, keys: list ):
        for key in keys:
            vols = sample_dict[key]
            vol_tensors = []
            for vol in vols:
                vol_tensor = torch.from_numpy(vol).type(torch.FloatTensor)
                vol_tensors.append(vol_tensor)
            sample_dict[key] = vol_tensors
        return sample_dict



class OpFixProstateBSequence(OpBase):

    def __call__(self, sample_dict: NDict, op_id: Optional[str],
                 key_sequence_ids: str, key_path_prefix:str, key_in_volumes_prefix:str):
        seq_ids = sample_dict[key_sequence_ids]
        if 'b' in seq_ids:

            B_SER_FIX = ['diffusie-3Scan-4bval_fs',
                         'ep2d_DIFF_tra_b50_500_800_1400_alle_spoelen',
                         'diff tra b 50 500 800 WIP511b alle spoelen']

            def get_single_item(a):
                if isinstance(a, list):
                    assert len(a)==1
                    return a[0]
                return a
            b_path = get_single_item(sample_dict[f'{key_path_prefix}b'])

            if os.path.basename(b_path) in B_SER_FIX:
                adc_volume = get_single_item(sample_dict[f'{key_in_volumes_prefix}ADC'])

                for b_seq_id in ['b800', 'b400']:
                    volume = get_single_item(sample_dict[f'{key_in_volumes_prefix}{b_seq_id}'])

                    volume.CopyInformation(adc_volume)
        return sample_dict

######################################3


class OpDeleteSequences(OpBase):
    def __init__(self, sequences_to_delete, **kwargs):
        super().__init__(**kwargs)
        self._sequences_to_delete = sequences_to_delete

    def __call__(self, sample_dict: NDict, op_id: Optional[str], key_sequence_ids):
        for seq_id in self._sequences_to_delete:
            _delete_seqeunce_from_dict(seq_id=seq_id, sample_dict=sample_dict, key_sequence_ids=key_sequence_ids)


def _delete_seqeunce_from_dict(seq_id, sample_dict, key_sequence_ids):
    seq_ids = sample_dict[key_sequence_ids]
    if seq_id in seq_ids:
        seq_ids.remove(seq_id)
        keys_to_delete = [k for k in sample_dict.flatten() if k.endswith(f'.{seq_id}')]
        for key in keys_to_delete:
            del sample_dict[key]


############################

def get_zeros_vol(vol):
    if vol.GetNumberOfComponentsPerPixel() > 1:
        ref_zeros_vol = sitk.VectorIndexSelectionCast(vol, 0)
    else:
        ref_zeros_vol = vol
    zeros_vol = np.zeros_like(sitk.GetArrayFromImage(ref_zeros_vol))
    zeros_vol = sitk.GetImageFromArray(zeros_vol)
    zeros_vol.CopyInformation(ref_zeros_vol)
    return zeros_vol




def crop_lesion_vol_mask_based(vol:sitk.sitkFloat32, position:tuple, ref:sitk.sitkFloat32, size:Tuple[int,int,int]=(160, 160, 32),
                    spacing:Tuple[int,int,int]=(1, 1, 3), mask_inx = -1,is_use_mask=True):
    """
    crop_lesion_vol crop tensor around position
    :param vol: vol to crop
    :param position: point to crop around
    :param ref: reference volume
    :param size: size in pixels to crop
    :param spacing: spacing to resample the col
    :param center_slice: z coordinates of position
    :param mask_inx: channel index in which mask is located default: last channel
    :param is_use_mask: use mask to define crop bounding box
    :return: cropped volume
    """

    margin = [20,20,0]
    vol_np = sitk.GetArrayFromImage(vol)
    if is_use_mask:

        mask = sitk.GetArrayFromImage(vol)[:,:,:,mask_inx]
        mask_bool = np.zeros(mask.shape).astype(int)
        mask_bool[mask>0.01]=1
        mask_final = sitk.GetImageFromArray(mask_bool)
        mask_final.CopyInformation(ref)

        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(mask_final)
        bounding_box = np.array(lsif.GetBoundingBox(1))
        vol_np[:, :, :, mask_inx] = mask_bool
    else:
        bounding_box = np.array([int(position[0]) - int(size[0] / 2),
                       int(position[1]) - int(size[1] / 2),
                       int(position[2]) - int(size[2] / 2),
                       size[0],
                       size[1],
                       size[2]
                       ])
    # in z use a fixed number of slices,based on position
    bounding_box[-1] = size[2]
    bounding_box[2] = int(position[2]) - int(size[2]/2)

    bounding_box_size = bounding_box[3:5][np.argmax(bounding_box[3:5])]
    dshift = bounding_box[3:5] - bounding_box_size
    dshift = np.append(dshift,0)

    ijk_min_bound = np.maximum(bounding_box[0:3]+dshift - margin,0)
    ijk_max_bound = np.maximum(bounding_box[0:3]+dshift+[bounding_box_size,bounding_box_size,bounding_box[-1]] + margin,0)



    vol_np_cropped = vol_np[ijk_min_bound[2]:ijk_max_bound[2],ijk_min_bound[1]:ijk_max_bound[1],ijk_min_bound[0]:ijk_max_bound[0],:]
    vol_np_resized = np.zeros((size[2],size[0],size[1],vol_np_cropped.shape[-1]))
    for si in range(vol_np_cropped.shape[0]):
        for ci in range(vol_np_cropped.shape[-1]):
            vol_np_resized[si,:,:,ci] = cv2.resize(vol_np_cropped[si, :,:, ci], (size[0],size[1]), interpolation=cv2.INTER_AREA)

    img = sitk.GetImageFromArray(vol_np_resized)
    return img




def extract_mask_from_annotation(vol_ref, bbox_coords):
    xstart = bbox_coords[0]
    ystart = bbox_coords[1]
    zstart = bbox_coords[2]
    xsize = bbox_coords[3]
    ysize = bbox_coords[4]
    zsize = bbox_coords[5]

    mask = get_zeros_vol(vol_ref)
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np[zstart:zstart + zsize, ystart:ystart + ysize, xstart:xstart + xsize] = 1.0
    return mask_np


def apply_rescaling(img: np.array, thres: tuple = (1.0, 99.0), method: str = 'noclip'):
    """
    apply_rescaling rescale each channal using method
    :param img:
    :param thres:
    :param method:
    :return:
    """
    eps = 0.000001

    def rescale_single_channel_image(img):
        # Deal with negative values first
        min_value = np.min(img)
        if min_value < 0:
            img -= min_value
        if method == 'clip':
            val_l, val_h = np.percentile(img, thres)
            img2 = img
            img2[img < val_l] = val_l
            img2[img > val_h] = val_h
            img2 = (img2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        elif method == 'mean':
            img2 = img / max(np.mean(img), 1)
        elif method == 'median':
            img2 = img / max(np.median(img), 1)
            # write as op
            ######################
        elif method == 'noclip':
            val_l, val_h = np.percentile(img, thres)
            img2 = img
            img2 = (img2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        else:
            img2 = img
        return img2

    # fix outlier image values
    img[np.isnan(img)] = 0
    # Process each channel independently
    if len(img.shape) == 4:
        for i in range(img.shape[-1]):
            img[..., i] = rescale_single_channel_image(img[..., i])
    else:
        img = rescale_single_channel_image(img)

    return img


def extract_seq_2_info_map(sample_path, series_desc_2_sequence_map):
    seq_info_dict = {}
    for seq_dir in os.listdir(sample_path):
        seq_path = os.path.join(sample_path, seq_dir)
        # read series description from dcm files
        dcm_files = glob.glob(os.path.join(seq_path,'*.dcm'))
        dcm_ds = pydicom.dcmread(dcm_files[0])

        series_desc = pydicom.dcmread(dcm_files[0]).SeriesDescription
        seq_id = series_desc_2_sequence_map.get(series_desc, 'UNKNOWN')

        series_num = extract_ser_num(dcm_ds)
        dicom_field = extract_dicom_field(dcm_ds, seq_id)


        seq_info_dict.setdefault(seq_id, []).append(dict(path=seq_path, series_num=series_num, dicom_field=dicom_field))

    return seq_info_dict

def extract_ser_num(dcm_ds):

    # series number
    if hasattr(dcm_ds, 'AcquisitionNumber'):
        return int(dcm_ds.AcquisitionNumber)
    return int(dcm_ds.SeriesNumber)

def extract_dicom_field(dcm_ds, seq_desc):

    # dicom key
    if seq_desc == 'b_mix':
        if 'DiffusionBValue' in dcm_ds:
            dicom_field = (0x0018, 0x9087)  # 'DiffusionBValue'
        else:
            dicom_field = (0x19, 0x100c)
    elif 'DCE' in seq_desc:
        if 'TemporalPositionIdentifier' in dcm_ds:
            dicom_field = (0x0020, 0x0100)  # Temporal Position Identifier
        elif 'TemporalPositionIndex' in dcm_ds:
            dicom_field = (0x0020, 0x9128)
        else:
            dicom_field = (0x0020, 0x0012)  # Acqusition Number
    elif seq_desc == 'MASK':
        dicom_field = (0x0020, 0x0011)  # series number
    else:
        dicom_field = 'NAN'

    return dicom_field

def sort_dicoms_by_field(seq_path, dicom_field, use_order_indicator):
    '''
    Return location dir of requested sequence
    '''

    """
     sort_dicom_by_dicom_field sorts the dcm_files based on dicom_field
     For some MRI sequences different kinds of MRI series are mixed together (as in bWI) case
     This function creates a dict={dicom_field_type:list of relevant dicoms},
     than concats all to a list of the different series types
     :param dcm_files: list of all dicoms , mixed
     :param dicom_field: dicom field to sort based on
     :return: sorted_names_list, list of sorted dicom series
     """
    dcm_files = glob.glob(os.path.join(seq_path, '*.dcm'))
    dcm_values = {}
    dcm_patient_z = {}
    dcm_instance = {}
    for index, dcm in enumerate(dcm_files):
        dcm_ds = pydicom.dcmread(dcm)
        patient_z = int(dcm_ds.ImagePositionPatient[2])
        instance_num = int(dcm_ds.InstanceNumber)
        try:
            val = int(dcm_ds[dicom_field].value)
            if val not in dcm_values:
                dcm_values[val] = []
                dcm_patient_z[val] = []
                dcm_instance[val] = []
            dcm_values[val].append(os.path.split(dcm)[-1])
            dcm_patient_z[val].append(patient_z)
            dcm_instance[val].append(instance_num)
        except:
            # sort by
            if index == 0:
                patient_z_ = []
                for dcm_ in dcm_files:
                    dcm_ds_ = pydicom.dcmread(dcm_)
                    patient_z_.append(dcm_ds_.ImagePositionPatient[2])
            val = int(np.floor((instance_num - 1) / len(np.unique(patient_z_))))
            if val not in dcm_values:
                dcm_values[val] = []
                dcm_patient_z[val] = []
                dcm_instance[val] = []
            dcm_values[val].append(os.path.split(dcm)[-1])
            dcm_patient_z[val].append(patient_z)
            dcm_instance[val].append(instance_num)

    sorted_keys = np.sort(list(dcm_values.keys()))
    sorted_names_list = [dcm_values[key] for key in sorted_keys]
    dcm_patient_z_list = [dcm_patient_z[key] for key in sorted_keys]
    dcm_instance_list = [dcm_instance[key] for key in sorted_keys]

    if use_order_indicator:
        # sort from low patient z to high patient z
        sorted_names_list_ = [list(np.array(list_of_names)[np.argsort(list_of_z)]) for list_of_names, list_of_z in
                              zip(sorted_names_list, dcm_patient_z_list)]
    else:
        # sort by instance number
        sorted_names_list_ = [list(np.array(list_of_names)[np.argsort(list_of_z)]) for list_of_names, list_of_z in
                              zip(sorted_names_list, dcm_instance_list)]

    return sorted_names_list_
def _get_as_list(x):
    if isinstance(x, list):
        return x
    return [x]