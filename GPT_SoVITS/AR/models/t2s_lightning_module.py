# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_lightning_module.py
# reference: https://github.com/lifeiteng/vall-e
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
from typing import Dict

import torch
from pytorch_lightning import LightningModule

from AR.models.t2s_model import Text2SemanticDecoder
from AR.modules.lr_schedulers import WarmupCosineLRSchedule
from AR.modules.optim import ScaledAdam

# TPU 지원을 위한 유틸리티 함수 가져오기
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils_tpu import is_tpu_available, get_xla_device, move_to_device, get_device_type, create_xla_optimizer


class Text2SemanticLightningModule(LightningModule):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        
        # 디바이스 설정
        self.device_type = get_device_type()
        if self.device_type == "tpu":
            print("TPU 디바이스를 사용하여 모델을 초기화합니다")
            self.device = get_xla_device()
        elif torch.cuda.is_available():
            print("CUDA 디바이스를 사용하여 모델을 초기화합니다")
            self.device = torch.device("cuda")
        else:
            print("CPU를 사용하여 모델을 초기화합니다")
            self.device = torch.device("cpu")
            
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        pretrained_s1 = config.get("pretrained_s1")
        if pretrained_s1 and is_train:
            # print(self.load_state_dict(torch.load(pretrained_s1,map_location="cpu")["state_dict"]))
            print(
                self.load_state_dict(
                    torch.load(
                        pretrained_s1,
                        map_location="cpu",
                    )["weight"],
                )
            )
        
        # 모델을 적절한 디바이스로 이동
        self.model = move_to_device(self.model, self.device)
            
        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        forward = self.model.forward if self.config["train"].get("if_dpo", False) == True else self.model.forward_old
        
        # TPU 환경에서는 데이터를 XLA 디바이스로 이동
        if self.device_type == "tpu":
            batch = move_to_device(batch, self.device)
            
        loss, acc = forward(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )
        self.manual_backward(loss)
        
        # TPU에서는 최적화 단계를 다르게 처리
        if batch_idx > 0 and batch_idx % 4 == 0:
            if self.device_type == "tpu":
                import torch_xla.core.xla_model as xm
                xm.optimizer_step(opt)
                opt.zero_grad()
                scheduler.step()
            else:
                opt.step()
                opt.zero_grad()
                scheduler.step()

        self.log(
            "total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"top_{self.top_k}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def validation_step(self, batch: Dict, batch_idx: int):
        return

    # # get loss
    # loss, acc = self.model.forward(
    #     batch['phoneme_ids'], batch['phoneme_ids_len'],
    #     batch['semantic_ids'], batch['semantic_ids_len'],
    #     batch['bert_feature']
    # )
    #
    # self.log(
    #     "val_total_loss",
    #     loss,
    #     on_step=True,
    #     on_epoch=True,
    #     prog_bar=True,
    #     sync_dist=True)
    # self.log(
    #     f"val_top_{self.top_k}_acc",
    #     acc,
    #     on_step=True,
    #     on_epoch=True,
    #     prog_bar=True,
    #     sync_dist=True)
    #
    # # get infer output
    # semantic_len = batch['semantic_ids'].size(1)
    # prompt_len = min(int(semantic_len * 0.5), 150)
    # prompt = batch['semantic_ids'][:, :prompt_len]
    # pred_semantic = self.model.infer(batch['phoneme_ids'],
    #                                  batch['phoneme_ids_len'], prompt,
    #                                  batch['bert_feature']
    #                                  )
    # save_name = f'semantic_toks_{batch_idx}.pt'
    # save_path = os.path.join(self.eval_dir, save_name)
    # torch.save(pred_semantic.detach().cpu(), save_path)

    def configure_optimizers(self):
        model_parameters = self.model.parameters()
        parameters_names = []
        parameters_names.append([name_param_pair[0] for name_param_pair in self.model.named_parameters()])
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
        
        # TPU 환경에서 최적화된 옵티마이저 생성
        if self.device_type == "tpu":
            lm_opt = create_xla_optimizer(lm_opt)
            print("TPU에 최적화된 옵티마이저를 생성했습니다")

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.config["optimizer"]["lr_init"],
                    peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"],
                    warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"],
                )
            },
        }
