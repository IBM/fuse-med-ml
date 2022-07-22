from typing import List, Union

from torch import Tensor

from fuseimg.data.ops.aug import geometry

from fuse.utils.ndict import NDict

from fuse.data import OpBase


class OpRotation3D(OpBase):
    """
    2D affine transformation
    """

    def __init__(self, verify_arguments: bool = True):
        """
        :param verify_arguments: this op expects torch tensor with either 2 or 3 dimensions. Set to False to disable verification
        """
        super().__init__()
        self._verify_arguments = verify_arguments

    def __call__(
        self, sample_dict: NDict, key: str, ax1_rot: float = 0.0, ax2_rot: float = 0.0, ax3_rot: float = 0
    ) -> Union[None, dict, List[dict]]:
        aug_input = sample_dict[key]

        aug_tensor = rotation_in_3d(aug_input, ax1_rot=ax1_rot, ax2_rot=ax2_rot, ax3_rot=ax3_rot)
        sample_dict[key] = aug_tensor
        return sample_dict


def rotation_in_3d(aug_input: Tensor, ax1_rot: float = 0.0, ax2_rot: float = 0.0, ax3_rot: float = 0):
    """
    rotates an input tensor around an axis, when for example z_rot is chosen,
    the rotation is in the x-y plane.
    Note: rotation angles are in relation to the original axis (not the rotated one)
    rotation angles should be given in degrees
    :param aug_input:image input should be in shape [channel, ax1, ax2, ax3]
    :param ax1_rot: angle to rotate ax2-ax3 plane clockwise
    :param ax2_rot: angle to rotate ax3-ax1 plane clockwise
    :param ax3_rot: angle to rotate ax1-ax2 plane clockwise
    :return:
    """
    assert len(aug_input.shape) == 4  # will only work for 3d
    channels = aug_input.shape[0]
    if ax1_rot != 0:
        squeez_img = geometry.squeeze_3D_to_2D(aug_input, axis_squeeze=1)
        rot_squeeze = geometry.auf_affine_2D(squeez_img, rotate=ax1_rot)
        aug_input = geometry.unsqueeze_3D_from_2D(rot_squeeze, axis_squeeze=1, channels=channels)
    if ax2_rot != 0:
        squeez_img = geometry.squeeze_3D_to_2D(aug_input, axis_squeeze=2)
        rot_squeeze = geometry.auf_affine_2D(squeez_img, rotate=ax2_rot)
        aug_input = geometry.unsqueeze_3D_from_2D(rot_squeeze, axis_squeeze=2, channels=channels)
    if ax3_rot != 0:
        squeez_img = geometry.squeeze_3D_to_2D(aug_input, axis_squeeze=3)
        rot_squeeze = geometry.auf_affine_2D(squeez_img, rotate=ax3_rot)
        aug_input = geometry.unsqueeze_3D_from_2D(rot_squeeze, axis_squeeze=3, channels=channels)

    return aug_input
