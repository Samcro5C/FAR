import importlib
from copy import deepcopy
from os import path as osp

from far.utils.misc import scandir
from far.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'far.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    if dataset_opt['type'] == 'custom_pixels':
        from hydra.utils import instantiate
        base_ds = instantiate(dataset_opt["hydra_dataset"])
        max_samples = dataset_opt.get("max_samples", None)
        if max_samples is not None:
            import numpy as np
            from torch.utils.data import Subset
            max_samples = int(max_samples)
            subset_seed = int(dataset_opt.get("subset_seed", 12345))  # same across ranks
            rng = np.random.default_rng(subset_seed)
            idx = rng.choice(len(base_ds), size=min(max_samples, len(base_ds)), replace=False)
            base_ds = Subset(base_ds, idx.tolist())
            base_ds.timestamps = base_ds.dataset.timestamps[base_ds.indices]

        from .custom_adapter import CustomSequenceAdapter
        ds = CustomSequenceAdapter(
            base_dataset=base_ds,
            context_frames=dataset_opt.get('context_frames'),
            predict_frames=dataset_opt.get('predict_frames'),
        )
        return ds
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    return dataset
