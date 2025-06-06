# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/data_module.py
# reference: https://github.com/lifeiteng/vall-e
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from AR.data.bucket_sampler import DistributedBucketSampler
from AR.data.dataset import Text2SemanticDataset

# TPU 지원을 위한 유틸리티 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils_tpu import is_tpu_available, get_xla_device, create_parallel_loader, create_tpu_data_sampler, sync_tpu_cores


class Text2SemanticDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        train_semantic_path,
        train_phoneme_path,
        dev_semantic_path=None,
        dev_phoneme_path=None,
    ):
        super().__init__()
        self.config = config
        self.train_semantic_path = train_semantic_path
        self.train_phoneme_path = train_phoneme_path
        self.dev_semantic_path = dev_semantic_path
        self.dev_phoneme_path = dev_phoneme_path
        self.num_workers = self.config["data"]["num_workers"]

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        self._train_dataset = Text2SemanticDataset(
            phoneme_path=self.train_phoneme_path,
            semantic_path=self.train_semantic_path,
            max_sec=self.config["data"]["max_sec"],
            pad_val=self.config["data"]["pad_val"],
        )
        self._dev_dataset = self._train_dataset
        # self._dev_dataset = Text2SemanticDataset(
        #     phoneme_path=self.dev_phoneme_path,
        #     semantic_path=self.dev_semantic_path,
        #     max_sample=self.config['data']['max_eval_sample'],
        #     max_sec=self.config['data']['max_sec'],
        #     pad_val=self.config['data']['pad_val'])

    def train_dataloader(self):
        batch_size = (
            self.config["train"]["batch_size"] // 2
            if self.config["train"].get("if_dpo", False) is True
            else self.config["train"]["batch_size"]
        )
        batch_size = max(min(batch_size, len(self._train_dataset) // 4), 1)  # 防止不保存
        
        # TPU 환경에서는 TPU에 최적화된 샘플러 사용
        if is_tpu_available():
            sampler = create_tpu_data_sampler(self._train_dataset, batch_size=batch_size)
        else:
            sampler = DistributedBucketSampler(self._train_dataset, batch_size=batch_size)
        
        dataloader = DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=16,
        )
        
        # TPU 환경에서는 병렬 로더로 변환
        if is_tpu_available():
            device = get_xla_device()
            return create_parallel_loader(dataloader, device)
        
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
            num_workers=max(self.num_workers, 12),
            persistent_workers=True,
            prefetch_factor=16,
        )
        
        # TPU 환경에서는 병렬 로더로 변환
        if is_tpu_available():
            device = get_xla_device()
            return create_parallel_loader(dataloader, device)
            
        return dataloader

    # 这个会使用到嘛？
    def test_dataloader(self):
        dataloader = DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
        )
        
        # TPU 환경에서는 병렬 로더로 변환
        if is_tpu_available():
            device = get_xla_device()
            return create_parallel_loader(dataloader, device)
            
        return dataloader
