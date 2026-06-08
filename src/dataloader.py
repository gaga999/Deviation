import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

# 嘗試引入 fastmri 套件
try:
    import fastmri
    from fastmri.data import mri_data
    from fastmri.data import transforms as fastmri_transforms
    from fastmri.data.subsample import RandomMaskFunc
except ImportError:
    print("Warning: 'fastmri' package is not installed. FastMRIDataset will not work.")


# =====================================================================
# 1. LoDoPaB-CT Dataset
# =====================================================================
class LoDoPaBDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        :param data_dir: 存放 LoDoPaB HDF5 檔案的目錄
        :param mode: 'train', 'validation', 或 'test'
        """
        self.observation_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('obs') and mode in f])
        self.ground_truth_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('gt') and mode in f])
        
        if len(self.observation_files) == 0:
            raise FileNotFoundError(f"No LoDoPaB {mode} files found in {data_dir}")

        # LoDoPaB 每個 HDF5 檔案固定有 128 個樣本
        self.samples_per_file = 128 
        self.total_samples = len(self.observation_files) * self.samples_per_file

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        
        obs_path = self.observation_files[file_idx]
        gt_path = self.ground_truth_files[file_idx]
        
        with h5py.File(obs_path, 'r') as f_obs, h5py.File(gt_path, 'r') as f_gt:
            y = f_obs['data'][sample_idx]
            x_true = f_gt['data'][sample_idx]
            
        # 轉成 PyTorch 張量並加上 Channel 維度 -> (1, H, W)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        x_true = torch.tensor(x_true, dtype=torch.float32).unsqueeze(0)
        
        # x0 作為演算法初始猜測，我們用全零張量
        x0 = torch.zeros_like(x_true) 
        
        return x0, y, x_true


# =====================================================================
# 2. fastMRI Dataset (基於官方 SliceDataset)
# =====================================================================
class UnrolledOptimizationTransform:
    """
    這是一個自定義的轉換器，用來橋接 fastMRI 的資料格式與我們的 Unrolled Optimizer
    """
    def __init__(self, mask_func):
        self.mask_func = mask_func

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        當 SliceDataset 讀到一張切片時，會將原始資料丟進這個函數
        """
        # 1. 將 numpy k-space 轉成 PyTorch 張量 (H, W, 2)
        kspace_torch = fastmri_transforms.to_tensor(kspace)
        
        # 2. 套用欠採樣遮罩
        # apply_mask 會回傳 masked_kspace, mask_本身, 以及套用的形狀
        masked_kspace, _, _ = fastmri_transforms.apply_mask(kspace_torch, self.mask_func)
        
        # 3. 處理 Target (真實影像)
        target_torch = fastmri_transforms.to_tensor(target)
        
        # 4. 加上 Channel 維度以符合我們 NCHW 的架構
        # masked_kspace: (1, H, W, 2), x_true: (1, H, W)
        y = masked_kspace.unsqueeze(0) 
        x_true = target_torch.unsqueeze(0)
        
        # 5. 初始猜測 x0
        x0 = torch.zeros_like(x_true)
        
        return x0, y, x_true


def get_fastmri_dataloader(data_dir, batch_size, center_fractions=[0.08], accelerations=[4], mode='train'):
    """
    幫助你快速建立 fastMRI DataLoader 的輔助函數
    """
    # 建立隨機遮罩 (例如：保留中心 8% 的低頻，整體加速 4 倍)
    mask_func = RandomMaskFunc(center_fractions=center_fractions, accelerations=accelerations)
    
    # 初始化我們的轉換器
    transform = UnrolledOptimizationTransform(mask_func)
    
    # 使用 fastMRI 官方的 SliceDataset
    # challenge='singlecoil' 代表我們處理單線圈資料
    dataset = mri_data.SliceDataset(
        root=data_dir,
        transform=transform,
        challenge='singlecoil'
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(mode == 'train'), 
        num_workers=4
    )
    return loader