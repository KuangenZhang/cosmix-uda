import os
import pickle
import numpy as np
import glob
from torch.utils.data import Dataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class BaseDataset(Dataset):
    def __init__(self,
                 version: str,
                 phase: str,
                 dataset_path: str,
                 voxel_size: float = 0.05,
                 sub_num: int = 50000,
                 use_intensity: bool = False,
                 augment_data: bool = False,
                 num_classes: int = 7,
                 ignore_label: int = None,
                 device: str = None,
                 weights_path: str = None):

        self.CACHE = {}
        self.version = version
        self.phase = phase
        self.dataset_path = dataset_path
        self.voxel_size = voxel_size  # in meter
        self.sub_num = sub_num
        self.use_intensity = use_intensity
        self.augment_data = augment_data and self.phase == 'train'
        self.num_classes = num_classes

        self.ignore_label = ignore_label
        
        # for input augs
        # self.clip_bounds = ((-100, 100), (-100, 100), (-100, 100))
        self.clip_bounds = None
        self.scale_augmentation_bound = (0.95, 1.05)
        self.rotation_augmentation_bound = ((-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20))
        self.translation_augmentation_ratio_bound = None

        self.device = device

        self.split = {'train': [],
                      'validation': []}

        self.maps = None
        self.color_map = None

        self.weights_path = weights_path
        if self.weights_path is not None:
            self.weights = np.load(self.weights_path)
        else:
            self.weights = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i: int):
        raise NotImplementedError

    def random_sample(self, points: np.ndarray, center: np.array = None) -> np.array:
        """
        :param points: input points of shape [N, 3]
        :param center: center to sample around, default is None, not used for now
        :return: np.ndarray of N' points sampled from input points
        """

        num_points = points.shape[0]

        if self.sub_num is not None:
            if self.sub_num <= num_points:
                sampled_idx = np.random.choice(np.arange(num_points), self.sub_num, replace=False)
            else:
                over_idx = np.random.choice(np.arange(num_points), self.sub_num - num_points, replace=False)
                sampled_idx = np.concatenate([np.arange(num_points), over_idx])
        else:
            sampled_idx = np.arange(num_points)

        return sampled_idx
    
    def get_frame_nums(self, sequence: str):
        file_list = glob.glob(os.path.join(self.dataset_path, sequence, 'labels', '*.label'))
        # convert file names to frame numbers
        frame_nums = [int(os.path.basename(file).split('.')[0]) for file in file_list]
        return frame_nums