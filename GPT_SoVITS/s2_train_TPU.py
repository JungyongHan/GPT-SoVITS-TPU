import warnings

warnings.filterwarnings("ignore")
import os
import math
import utils

hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")

# TPU 지원 추가
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPT_SoVITS.utils_tpu import get_xla_device, move_to_device, optimize_tpu_memory, tpu_safe_mel_spectrogram
import logging

# TPUv4 최적화 설정
TPU_OPTIMIZED_KWARGS = {
    # 데이터 로더 최적화 설정
    'persistent_workers': True, # worker 가 데이터 로더를 계속 유지할지 여부
    'prefetch_factor': 2, # worker 가 미리 로드할 배치 수 (prefetch_factor * num_workers)
    'num_workers': 4, # 데이터 로더 worker 수

    # TPU 데이터 전송 최적화 설정
    'device_prefetch_size': 2, # TPU 에서 준비할 배치 수
    'loader_prefetch_size': 4, # CPU 에서 TPU에 보낼걸 미리 준비할 배치 수
    'host_to_device_transfer_threads': 4, # TPU에서 데이터 로더에서 호스트로 데이터를 전송하는 스레드 수
    
    # XLA 컴파일러 최적화 설정
    'static_shapes': True, # 정적 형태 사용 여부 - 컴파일 시간 단축 및 성능 향상
    'fixed_batch_size': True, # 고정된 배치 크기 사용 여부 - 재컴파일 방지
    'optimize_memory': True, # 메모리 최적화 사용 여부 - 메모리 사용량 감소
    'max_sequence_length': 1000, # 최대 시퀀스 길이 (패딩에 사용) - 정적 형태 유지
    'compile_mode': 'reduce-recompilations', # XLA 컴파일 모드 - 재컴파일 최소화
    
    # TPU v4 메모리 최적화 설정
    'gradient_accumulation_steps': 1, # 그래디언트 누적 단계 수 (메모리 부족 시 증가)
    'batch_size_divisor': 8, # 배치 크기 나눗셈 인자 (TPU 코어 수에 맞게 조정)
    'use_dynamic_shapes': False, # 동적 형태 사용 여부 (정적 형태가 더 효율적)
}

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from tqdm import tqdm
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from random import randint

from module import commons
from module.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss

from module.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from process_ckpt import savee

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 최저 정밀도로 속도 향상

global_step = 0

def main():
    # TPU 환경 설정
    os.environ['PJRT_DEVICE'] = 'TPU'
    
    import torch_xla
    print(f"TPU 멀티프로세싱 시작")
    debug_single_process = False
    torch_xla.launch(
        run, args=(1, hps), debug_single_process=debug_single_process)

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    
    # TPU 디바이스 설정
    import torch_xla.distributed.xla_backend
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm
    from torch_xla import runtime as xr

    device = xm.xla_device()
    dist.init_process_group('xla', init_method='xla://')

    # TPU 환경 설정
    n_gpus = xr.world_size()
    rank = xr.global_ordinal()
    print(f"XLA:OPENED {rank}/{n_gpus}")
    
    # 데이터셋 및 샘플러 설정
    train_dataset = TextAudioSpeakerLoader(hps.data, device_str=f"{device}:{rank}({n_gpus})")  
    
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [
            32, 300, 400, 500, 600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    collate_fn = TextAudioSpeakerCollate()
    
    # TPU에 최적화된 데이터 로더 설정
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        shuffle=False,
        num_workers=TPU_OPTIMIZED_KWARGS['num_workers'],
        persistent_workers=True,
        prefetch_factor=TPU_OPTIMIZED_KWARGS['prefetch_factor'],
        pin_memory=False
    )
    train_loader = pl.MpDeviceLoader(
        train_loader, 
        device,
        loader_prefetch_size=TPU_OPTIMIZED_KWARGS['loader_prefetch_size'],
        device_prefetch_size=TPU_OPTIMIZED_KWARGS['device_prefetch_size'],
        host_to_device_transfer_threads=TPU_OPTIMIZED_KWARGS['host_to_device_transfer_threads']
    )
    
    # 모델 생성
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    
    # TPU 메모리 최적화 적용
    if TPU_OPTIMIZED_KWARGS['optimize_memory']:
        optimize_tpu_memory()
    
    # XLA 컴파일러 최적화 설정 적용
    if TPU_OPTIMIZED_KWARGS['static_shapes']:
        import torch_xla.core.xla_opts as xo
        xo.set_replication_devices([])
        xo.set_default_device(None)
        xo.set_use_spmd(False)
        xo.set_use_virtual_device(False)
        xo.set_compile_mode(TPU_OPTIMIZED_KWARGS['compile_mode'])
    
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            print(name, "not requires_grad")

    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(
        lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
        net_g.parameters(),
    )

    effective_lr = hps.train.learning_rate
    effective_eps = hps.train.eps
    
    optim_g = AdamW(
        [
            {"params": base_params, "lr": effective_lr},
            {
                "params": net_g.enc_p.text_embedding.parameters(),
                "lr": effective_lr * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.encoder_text.parameters(),
                "lr": effective_lr * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.mrte.parameters(),
                "lr": effective_lr * hps.train.text_low_lr_rate,
            },
        ],
        effective_lr,
        betas=hps.train.betas,
        eps=effective_eps,
    )
    optim_d = AdamW(
        net_d.parameters(),
        effective_lr,
        betas=hps.train.betas,
        eps=effective_eps,
    )

    try:  # 체크포인트 로드 시도
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version), "D_*.pth"),
            net_d,
            optim_d,
        )
        if rank == 0:
            logger.info("loaded D")
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version), "G_*.pth"),
            net_g,
            optim_g,
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # 사전 학습 모델 로드
        epoch_str = 1
        global_step = 0
        if hps.train.pretrained_s2G != "" and hps.train.pretrained_s2G != None and os.path.exists(hps.train.pretrained_s2G):
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            print(
                "loaded pretrained %s" % hps.train.pretrained_s2G,
                net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                    strict=False,
                )
            )
        if (
            hps.train.pretrained_s2D != ""
            and hps.train.pretrained_s2D != None
            and os.path.exists(hps.train.pretrained_s2D)
        ):
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2D)
            print(
                "loaded pretrained %s" % hps.train.pretrained_s2D,
                net_d.load_state_dict(
                    torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"],
                ),
            )

    # 스케줄러 설정
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=hps.train.lr_decay,
        last_epoch=-1,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=hps.train.lr_decay,
        last_epoch=-1,
    )
    for _ in range(epoch_str):
        scheduler_g.step()
        scheduler_d.step()
    
    # TPU에서는 스케일러 사용 안함
    scaler = None

    xm.master_print(f"에포크 {epoch_str}부터 학습 시작")
        
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:               
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()
            
    xm.master_print("학습 완료")

def _get_device_spec(device):
    import torch_xla.runtime as xr
    ordinal = xr.global_ordinal()
    return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)

def _train_update(device, epoch, step, total_step, loss, tracker, writer):
    rate = tracker.rate()
    global_rate = tracker.global_rate()
    print(
        f"[Train] {_get_device_spec(device)} Epoch: [{epoch}/{hps.train.epochs}][{step}/{total_step}] "
        f"Loss: {loss.item():.4f} Rate: {rate:.4f} Global Rate: {global_rate:.4f} ",
        flush=True,
    )

def _save_checkpoint(net_g, net_d, optim_g, optim_d, hps, epoch):
    global global_step
    if hps.train.if_save_latest == 0:
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                "G_{}.pth".format(global_step),
            ),
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                "D_{}.pth".format(global_step),
            ),
        )
    else:
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                "G_{}.pth".format(233333333333),
            ),
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                "%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version),
                "D_{}.pth".format(233333333333),
            ),
        )

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
        
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    import torch_xla.debug.metrics_compare_utils as mcu
    import torch_xla.debug.metrics as met
    from torch_xla.amp import autocast
    
    xm.master_print(f"에포크 {epoch}: 학습 시작 (TPU 최적화 적용됨)")
    
    memory_error_count = 0
    max_memory_errors = 3
    tracker = None
    device = None
    
    try:
        for batch_idx, ( ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, ) in enumerate(train_loader):
            print(batch_idx)
            device = get_xla_device()
            tracker = xm.RateTracker()
            # 메모리 최적화: 한 번에 하나씩 텐서 이동 및 불필요한 참조 제거
            # 중복 이동 제거 - 각 텐서를 한 번만 이동
            spec = move_to_device(spec, device)
            spec_lengths = move_to_device(spec_lengths, device)
            y = move_to_device(y, device)
            y_lengths = move_to_device(y_lengths, device)
            text = move_to_device(text, device)
            text_lengths = move_to_device(text_lengths, device)
            # Keep ssl on CPU for now to save memory, will move just before use
            ssl.requires_grad = False
            # 정적 형태 유지를 위한 패딩 처리
            if TPU_OPTIMIZED_KWARGS['static_shapes']:
                # 시퀀스 길이 고정을 위한 패딩 처리
                max_len = TPU_OPTIMIZED_KWARGS['max_sequence_length']
                if spec.shape[2] < max_len:
                    pad_len = max_len - spec.shape[2]
                    spec = F.pad(spec, (0, pad_len), 'constant', 0)
            print("tensors moved to device with static shapes")

            with autocast(device=device, enabled=True):
                print("forward")
                # Move ssl to device just before use inside autocast
                ssl = move_to_device(ssl, device)
                (
                    y_hat,
                    kl_ssl,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    stats_ssl,
                ) = net_g(ssl, spec, spec_lengths, text, text_lengths)
                print("forward done")
                
                # TPU에 최적화된 mel spectrogram 생성
                mel = tpu_safe_mel_spectrogram(y.squeeze(1), hps.data)
                y_mel = tpu_safe_mel_spectrogram(y_hat.squeeze(1), hps.data)
                y = y.float()
                y_hat = y_hat.float()
                
                # 손실 계산
                y_dp_hat_r, y_dp_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                loss_mel = F.l1_loss(y_mel, y_hat) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_p, m_q, logs_q, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_dp_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                
                # 생성기 업데이트
                optim_g.zero_grad()
                loss_gen_all.backward()
                xm.optimizer_step(optim_g)
                
                # 판별기 업데이트
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_dp_hat_r, y_dp_hat_g)
                optim_d.zero_grad()
                loss_disc.backward()
                xm.optimizer_step(optim_d)
                
                # 메트릭 로깅
                if rank == 0 and batch_idx % hps.train.log_interval == 0:
                    scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc, "learning_rate": optim_g.param_groups[0]['lr'], "grad_norm_d": 0, "grad_norm_g": 0}
                    scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})

                    scalar_dict.update({"loss/g/{}" .format(i): v for i, v in enumerate(losses_gen)})
                    scalar_dict.update({"loss/d_r/{}" .format(i): v for i, v in enumerate(losses_disc_r)})
                    scalar_dict.update({"loss/d_g/{}" .format(i): v for i, v in enumerate(losses_disc_g)})
                    
                    image_dict = {"slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()), "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat[0].data.cpu().numpy()), "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())}
                    
                    utils.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict)
                
                # 진행 상황 업데이트
                if batch_idx % 5 == 0:
                    _train_update(device, epoch, batch_idx, len(train_loader), loss_gen_all, tracker, writer)
                
                xm.mark_step()
                global_step += 1
        # 체크포인트 저장
        if rank == 0 and global_step % hps.train.save_every_steps == 0 and global_step != 0:
            _save_checkpoint(net_g, net_d, optim_g, optim_d, hps, epoch)
                
                
                
    except Exception as e:
        print(f"Training error: {e}")
        memory_error_count += 1
        if memory_error_count >= max_memory_errors:
            print(f"Too many memory errors ({memory_error_count}), stopping training")
            return

    # 에포크 종료 후 체크포인트 저장
    if rank == 0:
        savee(global_step, net_g, net_d, optim_g, optim_d, hps.s2_ckpt_dir, hps.model.version)

if __name__ == "__main__":
    main()
