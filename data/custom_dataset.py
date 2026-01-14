import os
import yaml
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import List, Optional, Callable, Any, Dict, Tuple
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from data.base_dataset import BaseDataset, get_transform

def _open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert('RGB')


def to_paired_tensor(img_obj, transform_fn=None, is_output: bool = False):
    """
    Convert a PIL image / tensor / ndarray / path to a tensor matching PairedDataset conventions.

    - If transform_fn is provided it will be applied to a PIL Image input.
    - If input is a path (str/Path) the file will be opened as RGB.
    - Ensures output tensors are float and scaled so:
        * conditioning tensors are in [0,1]
        * output tensors are in [-1,1]
    """
    out = img_obj
    # If user passed a path, open it
    if isinstance(out, (str, Path)):
        out = _open_rgb_image(Path(out))

    # apply transform if available and input is PIL
    # NOTE: transform_fn should only be applied to PIL Images. If the caller
    # already applied the transform (i.e., passed a Tensor or ndarray), we
    # must NOT re-apply it here to avoid double preprocessing.
    if transform_fn is not None and isinstance(out, Image.Image):
        out = transform_fn(out)

    # convert to tensor if needed
    if isinstance(out, Image.Image):
        t = F.to_tensor(out)
    elif torch.is_tensor(out):
        t = out.clone()
    else:
        # try to coerce via numpy array -> PIL
        try:
            t = F.to_tensor(Image.fromarray(out))
        except Exception:
            raise TypeError(f"Unsupported transform output type: {type(out)}")
 
    return t


class PairsDataset(Dataset):
    """
    Dataset that loads good/bad image pairs from a YAML file.

    Each __getitem__ returns:
      - good_tensor, bad_tensor, info_dict

    Where transforms are applied separately to good/bad if provided.
    A postprocess_hook can compute pseudo-labels or additional metadata using PIL image or tensors.
    """

    def __init__(
        self,
        yaml_path: str,
        root_dir: Optional[str] = None,
        transform_good: Optional[Callable[[Image.Image], Any]] = None,
        transform_bad: Optional[Callable[[Image.Image], Any]] = None,
        sample_mode: str = 'random',
        seed: Optional[int] = None,
        test_mode=False, 
        postprocess_hook: Optional[Callable[[Any, str, bool], Dict[str, Any]]] = None,
        tokenizer: Optional[Any] = None,
        default_caption: Optional[str] = None,
    ) -> None:
        """
        Args:
            yaml_path: Path to YAML file with the given schema.
            root_dir: Optional root directory to resolve image paths; if None, use YAML file's parent.
            transform_good: Transform applied to good image (PIL -> tensor or other), optional.
            transform_bad: Transform applied to bad image, optional. Defaults to transform_good if None.
            sample_mode: 'random' or 'round_robin' to select among multiple bads.
            seed: Optional RNG seed for deterministic sampling.
            postprocess_hook: Optional function(image, path, is_good) -> dict; executed after transform.
        """
        self.yaml_path = Path(yaml_path)
        self.root_dir = Path(root_dir) if root_dir is not None else self.yaml_path.parent
        self.transform_good = transform_good
        self.transform_bad = transform_bad if transform_bad is not None else transform_good
        self.sample_mode = sample_mode
        self.postprocess_hook = postprocess_hook
        self.test_mode = test_mode 
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or 'pairs' not in data:
            raise ValueError(f"Invalid YAML format: top-level 'pairs' key not found in {yaml_path}")

        pairs = data['pairs'] or []
        self._entries = []
        # store captions and tokenized ids per entry (parallel to _entries)
        self._captions = []
        self._input_ids = []
        self._tokenizer = tokenizer
        self._default_caption = default_caption

        for item in pairs:
            good_rel = item.get('good')
            bad_list = item.get('bads', [])
            if not good_rel or not bad_list:
                continue
            good_path = str(self.root_dir / good_rel)
            bad_paths = []
            for bad in bad_list:
                # bad could be a dict with key 'file' or a direct string, handle both
                if isinstance(bad, dict):
                    bad_rel = bad.get('file')
                else:
                    bad_rel = str(bad)
                if not bad_rel:
                    continue
                bad_paths.append(str(self.root_dir / bad_rel))
            if len(bad_paths) == 0:
                continue
            # capture caption if present, else use dataset default or empty string
            caption = None
            # possible keys that may contain caption text
            for k in ('caption', 'captions', 'prompt', 'text'):
                if isinstance(item.get(k, None), str):
                    caption = item.get(k)
                    break
            if caption is None:
                caption = self._default_caption if self._default_caption is not None else ''

            # tokenize if tokenizer is provided
            input_ids = None
            if self._tokenizer is not None:
                # allow tokenizer to raise on invalid inputs so errors surface during debugging
                input_ids = self._tokenizer(
                    caption, max_length=self._tokenizer.model_max_length,
                    padding='max_length', truncation=True, return_tensors='pt'
                ).input_ids

            self._entries.append((good_path, bad_paths))
            self._captions.append(caption)
            self._input_ids.append(input_ids)

        if len(self._entries) == 0:
            raise ValueError(f"No valid pairs loaded from {yaml_path}")

        # For round-robin sampling maintain an index per entry
        if self.sample_mode == 'round_robin':
            self._rr_indices = [0 for _ in self._entries]
        else:
            self._rr_indices = None

    def __len__(self) -> int:
        return len(self._entries)

    def _select_bad(self, idx: int) -> str:
        good_path, bads = self._entries[idx]
        if self.sample_mode == 'random' or len(bads) == 1 or self._rr_indices is None:
            return bads[self._rng.randrange(len(bads))]
        # round robin
        rr = self._rr_indices[idx]
        selected = bads[rr]
        self._rr_indices[idx] = (rr + 1) % len(bads)
        return selected

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        good_path, _ = self._entries[idx]
        bad_path = self._select_bad(idx)

        good_img = _open_rgb_image(Path(good_path))
        if self.test_mode:   
            bad_img = good_img
        else:
            bad_img = _open_rgb_image(Path(bad_path))

        # 检查是否为同步变换，如果是则设置相同的随机状态
        if (hasattr(self.transform_good, '_set_random_state') and 
            hasattr(self.transform_bad, '_set_random_state')):
            # 生成随机状态
            random_state = {
                'random': random.getstate(),
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state()
            }
            
            # 设置相同的随机状态
            self.transform_good._set_random_state(random_state)
            self.transform_bad._set_random_state(random_state)

        # 同步随机参数（若有），然后应用变换
        # Apply transforms if they are provided. Transforms may return PIL or
        # already-converted tensors. We will detect that later and avoid
        # re-applying transforms in to_paired_tensor.
        good_out = self.transform_good(good_img) if self.transform_good is not None else good_img
        bad_out = self.transform_bad(bad_img) if self.transform_bad is not None else bad_img

        info: Dict[str, Any] = {
            'good_path': good_path,
            'bad_path': bad_path,
            'index': idx,
            'test_mode': self.test_mode,
        }

        # attach caption and tokenized ids if available
        try:
            caption_for_entry = self._captions[idx]
        except Exception:
            caption_for_entry = self._default_caption if self._default_caption is not None else ''
        info['caption'] = caption_for_entry
        try:
            input_ids_for_entry = self._input_ids[idx]
        except Exception:
            input_ids_for_entry = None
        info['input_ids'] = input_ids_for_entry
 
 
        out_t = to_paired_tensor(good_out, transform_fn=self.transform_good if isinstance(good_out, Image.Image) else None, is_output=True)
        cond_t = to_paired_tensor(bad_out, transform_fn=self.transform_bad if isinstance(bad_out, Image.Image) else None, is_output=False)
        out_t = F.normalize(out_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # scale to [-1,1] 
        # return Paired-style mapping
        return {
            "output_pixel_values": out_t,
            "conditioning_pixel_values": cond_t,
            "caption": info.get('caption'),
            "input_ids": info.get('input_ids'),
            "good_path": good_path,
            "bad_path": bad_path,
        }


class CompositeDataset(Dataset):
    """
    Composite dataset that mixes multiple homogeneous datasets (same output schema).

    You can provide per-dataset sampling weights, or it will sample uniformly across datasets by size.
    Each __getitem__ returns the sample from one of the underlying datasets.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        sampling_weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
        image_prep: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        reverseoutput: bool = False, 
    ) -> None:
        if not datasets:
            raise ValueError('CompositeDataset requires at least one dataset')
        self.datasets = datasets
        self._sizes = [len(d) for d in datasets]
        self._cum_sizes = []
        total = 0
        self.reverseoutput = reverseoutput 
        self.is_testmode = self.datasets[0].test_mode if hasattr(self.datasets[0],'test_mode') else False
        self.file_list=[] 
        for ds in datasets:
            ds_files=[] 
            for i in range(len(ds)):
                entry = ds._entries[i] if hasattr(ds,'_entries') else None
                if entry is not None:
                    good_path, bad_paths = entry
                    ds_files+= [good_path]+bad_paths
            self.file_list+= ds_files 
            
        for size in self._sizes:
            total += size
            self._cum_sizes.append(total)

        if sampling_weights is not None:
            if len(sampling_weights) != len(datasets):
                raise ValueError('sampling_weights length must match number of datasets')
            s = sum(sampling_weights)
            if s <= 0:
                raise ValueError('sampling_weights must sum to > 0')
            self._weights = [w / s for w in sampling_weights]
        else:
            # default: proportion by dataset size
            total_size = float(sum(self._sizes))
            self._weights = [sz / total_size for sz in self._sizes]

        self._rng = random.Random(seed) if seed is not None else random.Random()

        # Optional image preprocessing and tokenizer to normalize outputs to PairedDataset schema
        self.image_prep = image_prep
        self.tokenizer = tokenizer

        # Compute a nominal length: sum of sizes for a full epoch across datasets
        self._length = sum(self._sizes)

    def __len__(self) -> int:
        return self._length

    def _sample_dataset_index(self) -> int:
        r = self._rng.random()
        acc = 0.0
        for i, w in enumerate(self._weights):
            acc += w
            if r <= acc:
                return i
        return len(self._weights) - 1

    def __getitem__(self, idx: int):
        # ignore idx for stochastic mixing; optionally can map to per-dataset idx if desired
        ds_idx = self._sample_dataset_index()
        dataset = self.datasets[ds_idx]
        # sample an item index within that dataset uniformly
        item_idx = self._rng.randrange(len(dataset))
        sample = dataset[item_idx]
        if self.reverseoutput :
            sample["output_pixel_values"], sample["conditioning_pixel_values"] = sample["conditioning_pixel_values"], sample["output_pixel_values"]  
        sample['source_dataset'] = ds_idx
        
        return sample



def _instantiate_pairs_from_yaml_list(yamls, sample_mode='random', seed=None, transform=None, postprocess_hook=None, tokenizer=None, default_caption=None):
    """
    Given a list of dicts each containing 'path' and optional 'root', instantiate a list of PairsDataset
    and return them. Each entry in `yamls` is expected to be a mapping with keys 'path' and 'root'.
    """
    datasets = []
    for item in yamls:
        if not isinstance(item, dict):
            continue
        yaml_path = item.get('path')
        root = item.get('root', None)
        if yaml_path is None:
            continue
        ds = PairsDataset(
            yaml_path=str(yaml_path),
            root_dir=str(root) if root is not None else None,
            transform_good=transform,
            transform_bad=transform,
            sample_mode=sample_mode,
            seed=seed,
            postprocess_hook=postprocess_hook,
            tokenizer=tokenizer,
            default_caption=default_caption,
        )
        datasets.append(ds)
    return datasets


def build_dataset_from_config(dataset_cfg: dict, split: str = 'train', *, transform: Optional[Callable] = None, tokenizer: Optional[Any] = None, default_caption: Optional[str] = None):
    """
    Factory to construct a dataset from a config mapping (parsed from YAML/CLI config).
    """
    if not isinstance(dataset_cfg, dict):
        raise ValueError('dataset_cfg must be a dict')

    ds_type = dataset_cfg.get('type')

    # choose train/val sub-config
    split_cfg = dataset_cfg.get(split) or {}

    if ds_type == 'pairs_yaml':
        sample_mode = split_cfg.get('sample_mode', 'random')
        seed = split_cfg.get('seed', None)
        yamls = split_cfg.get('yamls', [])
        if not yamls:
            raise ValueError('pairs_yaml dataset requires yamls list in config')

        # instantiate PairsDataset objects
        pds = _instantiate_pairs_from_yaml_list(
            yamls, sample_mode=sample_mode, seed=seed, transform=transform,
            postprocess_hook=split_cfg.get('postprocess_hook', None),
            tokenizer=tokenizer,
            default_caption=dataset_cfg.get('default_caption', default_caption),
        )

        # extract optional sampling weights
        sampling_weights = split_cfg.get('sampling_weights', None)

        comp = CompositeDataset(
            datasets=pds,
            sampling_weights=sampling_weights,
            seed=seed,
            tokenizer=tokenizer,
            reverseoutput=dataset_cfg.get('reverseoutput', False) 
        )
        return comp

    return None

class _Adapter(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
        if len(base_ds) > 0:
            sample = self.base[0]   
            self.fixed_caption_src = sample.get('caption_src', "A underwater image")
            self.fixed_caption_tgt = sample.get('caption_tgt', "A clear, color-rich, colorcast-free, sharp-detail, naturally-lit, haze-free, high-contrast underwater image")
        else:
             self.fixed_caption_src = "A underwater image"
             self.fixed_caption_tgt = "A clear image"

        self.file_list = self.base.file_list if hasattr(self.base,'file_list') else None 
        if not self.file_list:
            # raise ValueError("Dataset does not have file_list attribute")  
            pass
            
    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        # If sample is dict in paired format
        if isinstance(sample, dict) and 'output_pixel_values' in sample and 'conditioning_pixel_values' in sample:
            out = sample['output_pixel_values']
            cond = sample['conditioning_pixel_values']
            # Ensure ranges: Paired returns output in [-1,1], conditioning in [0,1]
            # Unpaired training expects pixel_values_src and pixel_values_tgt both normalized to [-1,1]
            pv_src = cond.clone()
            # if pv_src.min() >= 0.0 and pv_src.max() <= 1.0:
            pv_src = pv_src * 2.0 - 1.0
            pv_tgt = out.clone()
            
            # out is already [-1,1] in Paired; leave
            return {
                'pixel_values_src': pv_src,
                'pixel_values_tgt': pv_tgt,
                'caption_src': sample.get('caption_src', self.fixed_caption_src),
                'caption_tgt': sample.get('caption', self.fixed_caption_tgt),
                'A': pv_src, # For CWR compatibility
                'B': pv_tgt, # For CWR compatibility
                'A_paths': sample.get('bad_path', ''),
                'B_paths': sample.get('good_path', ''),
            }
        # If sample is dict in unpaired format already, pass through
        if isinstance(sample, dict) and 'pixel_values_src' in sample and 'pixel_values_tgt' in sample:
            return sample
        # otherwise try to return as-is
        return sample


class CustomDataset(BaseDataset):
    """
    Adapter class to use the PairsDataset in this project.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--yaml_path', type=str, help='path to the yaml file')
        parser.add_argument('--sample_mode', type=str, default='random', choices=['random', 'round_robin'])
        parser.add_argument('--reverseoutput', action='store_true', help='reverse output and conditioning')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        yaml_path = getattr(opt, 'yaml_path', None) or opt.dataroot
        
        # We use convert=False because to_paired_tensor and PairsDataset handle conversion/normalization
        self.transform = get_transform(opt, convert=False)
        
        dataset_cfg = {
            'type': 'pairs_yaml',
            'train': {
                'yamls': [{'path': yaml_path}],
                'sample_mode': getattr(opt, 'sample_mode', 'random'),
            },
            'val': {
                'yamls': [{'path': yaml_path}],
                'sample_mode': getattr(opt, 'sample_mode', 'random'),
            },
            'reverseoutput': getattr(opt, 'reverseoutput', False)
        }
        
        split = 'train' if opt.isTrain else 'val'
        self.inner_ds = build_dataset_from_config(dataset_cfg, split=split, transform=self.transform)
        if self.inner_ds is None:
             raise ValueError("Failed to create inner dataset from config")
        self.adapter = _Adapter(self.inner_ds)

    def __getitem__(self, index):
        return self.adapter[index]

    def __len__(self):
        return len(self.adapter)
