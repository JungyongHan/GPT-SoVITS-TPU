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
from GPT_SoVITS.utils_tpu import is_tpu_available
import logging

# TPUv4 최적화 설정
TPU_OPTIMIZED_KWARGS = {
    'persistent_workers': False,
    'prefetch_factor': 16,
    'loader_prefetch_size': 8,
    'device_prefetch_size': 4,
    'num_workers': 0,
    'host_to_device_transfer_threads': 1,
}

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
# TPU 환경에서는 TPU 호환 버전의 mel_processing 모듈 사용

from module.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from process_ckpt import savee

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
###反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
# from config import pretrained_s2G,pretrained_s2D
global_step = 0

# 디바이스 설정

device = "cpu"  # cuda以外的设备，等mps优化后加入

def main():
    # TPU 또는 GPU 설정
    if is_tpu_available():
        os.environ['PJRT_DEVICE'] = 'TPU'
        from GPT_SoVITS.utils_tpu import get_tpu_cores_count
        num_cores = get_tpu_cores_count()  # 자동으로 TPU 코어 수 감지
        # num_cores = 1 # temp
    
        import torch_xla
        print(f"TPU 멀티프로세싱 시작 (코어 수: {num_cores})")
        debug_single_process = num_cores == 1
        torch_xla.launch(
            run, args=(num_cores, hps), debug_single_process=debug_single_process)
        return
    
    # GPU 또는 CPU 설정
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.s2_ckpt_dir)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    # TPU 환경인 경우 분산 설정 건너뛰기 (XLA가 처리)
    if not is_tpu_available():
        dist.init_process_group(
            backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
            init_method="env://?use_libuv=False",
            world_size=n_gpus,
            rank=rank,
        )
    else:
        # TPU 환경에서 필요한 추가 모듈 import
        import torch_xla.core.xla_model as xm
        dist.init_process_group('xla', init_method='xla://')
    
    torch.manual_seed(hps.train.seed)
    
    # 디바이스 설정
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()

    elif torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        # CPU 모드
        device = "cpu"

    # TPU v4-32에 최적화된 버킷 크기 조정 (더 작은 버킷으로 메모리 사용량 감소)
    train_dataset = TextAudioSpeakerLoader(hps.data)  ########
    
    # TPU v4-32에 최적화된 배치 크기 계산
    if is_tpu_available():
        effective_batch_size = max(1, hps.train.batch_size // 2)
        logging.info(f"TPU v4-32 환경에 최적화된 배치 크기 사용: {effective_batch_size}")
    else:
        effective_batch_size = hps.train.batch_size
    
    train_sampler = DistributedBucketSampler(
        train_dataset,
        effective_batch_size,
        [
            32,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
        ],
        num_replicas=n_gpus,
        rank=rank % n_gpus if is_tpu_available() else rank,
        shuffle=True,
    )

    collate_fn = TextAudioSpeakerCollate()
    # 데이터 로더 생성 - TPU에 최적화된 설정
    if is_tpu_available():
        train_loader = DataLoader(
            train_dataset,
            num_workers=TPU_OPTIMIZED_KWARGS['num_workers'],  # TPUv4 최적화 설정 적용
            shuffle=False,
            pin_memory=False,  # TPU에서는 pin_memory 사용 안함
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=TPU_OPTIMIZED_KWARGS['persistent_workers'],  # TPUv4 최적화 설정 적용
            prefetch_factor=TPU_OPTIMIZED_KWARGS['prefetch_factor'],  # TPUv4 최적화 설정 적용
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            num_workers=6,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=4,
        )
    
    # TPU용 병렬 로더 생성 (슬라이싱 환경에 최적화)
    if is_tpu_available():
        # 개선된 병렬 로더 함수 사용 (드롭 마지막 배치 옵션 추가)
        import torch_xla.distributed.parallel_loader as pl
        train_loader = pl.MpDeviceLoader(
            train_loader, 
            device,
            loader_prefetch_size=TPU_OPTIMIZED_KWARGS['loader_prefetch_size'],
            device_prefetch_size=TPU_OPTIMIZED_KWARGS['device_prefetch_size'],
            host_to_device_transfer_threads=TPU_OPTIMIZED_KWARGS['host_to_device_transfer_threads']
        )
        xm.master_print(f"TPUv4 최적화 병렬 로더 설정 완료: {TPU_OPTIMIZED_KWARGS}")
    # if rank == 0:
    #     eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, val=True)
    #     eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
    #                              batch_size=1, pin_memory=True,
    #                              drop_last=False, collate_fn=collate_fn)    
    # 모델 생성
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    
    # TPU v4-32에서 메모리 최적화 기법 적용
    
    # 적절한 디바이스로 모델 이동
    net_g = net_g.to(device)
    net_d = net_d.to(device)
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

    # te_p=net_g.enc_p.text_embedding.parameters()
    # et_p=net_g.enc_p.encoder_text.parameters()
    # mrte_p=net_g.enc_p.mrte.parameters()

    # TPU v4-32에 최적화된 옵티마이저 설정
    if is_tpu_available():
        # TPU에서는 더 작은 학습률과 더 큰 eps 값 사용
        effective_lr = hps.train.learning_rate * 0.8
        effective_eps = hps.train.eps * 10.0
        logging.info(f"TPU v4-32 환경에 최적화된 학습률 사용: {effective_lr}, eps: {effective_eps}")
    else:
        effective_lr = hps.train.learning_rate
        effective_eps = hps.train.eps
    
    optim_g = torch.optim.AdamW(
        # filter(lambda p: p.requires_grad, net_g.parameters()),###默认所有层lr一致
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
        # TPU에서 메모리 효율적인 옵티마이저 설정
        maximize=False,
        weight_decay=0.01 if is_tpu_available() else 0.0,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        effective_lr,
        betas=hps.train.betas,
        eps=effective_eps,
        # TPU에서 메모리 효율적인 옵티마이저 설정
        maximize=False,
        weight_decay=0.01 if is_tpu_available() else 0.0,
    )
    # 분산 학습 설정
    if is_tpu_available():
        # TPU는 DDP 대신 XLA의 자체 병렬 처리 사용
        pass
    elif torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version), "D_*.pth"),
            net_d,
            optim_d,
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s/logs_s2_%s" % (hps.data.exp_dir, hps.model.version), "G_*.pth"),
            net_g,
            optim_g,
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.train.pretrained_s2G != "" and hps.train.pretrained_s2G != None and os.path.exists(hps.train.pretrained_s2G):
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            print(
                "loaded pretrained %s" % hps.train.pretrained_s2G,
                net_g.module.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                    strict=False,
                )
                if torch.cuda.is_available()
                else net_g.load_state_dict(
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
                net_d.module.load_state_dict(
                    torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"],
                )
                if torch.cuda.is_available()
                else net_d.load_state_dict(
                    torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"],
                ),
            )

    net_g.to(device) 
    net_d.to(device)  

    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

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

    if is_tpu_available():
        scaler = None
    else:
        scaler = GradScaler(enabled=hps.train.fp16_run)

    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.master_print(f"에포크 {epoch_str}부터 학습 시작")
    else:
        print("start training from epoch %s" % epoch_str)
        
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
                # [train_loader, eval_loader], logger, [writer, writer_eval])
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
            
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.master_print("학습 완료")
    else:
        print("training done")

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
    # import sys
    # use sys.stdout.flush() instead of print()
    # sys.stdout.write(
    #     f"[Train] {_get_device_spec(device)} Epoch: [{epoch}/{hps.train.epochs}][{step}/{total_step}] "
    #     f"Loss: {loss.item():.4f} Rate: {rate:.4f} Global Rate: {global_rate:.4f} \n"
    # )
    # sys.stdout.flush()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
        
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.master_print(f"에포크 {epoch}: GAN 학습 준비 중...")
    else:
        print("preparing gan training")
        
    net_g.train()
    net_d.train()
    
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.master_print(f"에포크 {epoch}: 학습 시작")
    else:
        print("start training")
    # TPU v4-32에서 메모리 문제 방지를 위한 배치 처리 개선
    memory_error_count = 0
    max_memory_errors = 3
    tracker = None
    # TPU에서 XLA 컴파일러 최적화 설정
    
    try:
        for batch_idx, ( ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, ) in enumerate(train_loader):
            if is_tpu_available():
                from GPT_SoVITS.utils_tpu import move_to_device, get_xla_device
                device = get_xla_device()
                tracker = xm.RateTracker()
                # 메모리 최적화: 한 번에 하나씩 텐서 이동 및 불필요한 참조 제거
                spec = move_to_device(spec, device)
                spec_lengths = move_to_device(spec_lengths, device)
                y = move_to_device(y, device)
                y_lengths = move_to_device(y_lengths, device)
                ssl = move_to_device(ssl, device)
                ssl.requires_grad = False
                text = move_to_device(text, device)
                text_lengths = move_to_device(text_lengths, device)
                
                
            elif torch.cuda.is_available():
                spec, spec_lengths = (
                    spec.cuda(
                        rank,
                        non_blocking=True,
                    ),
                    spec_lengths.cuda(
                        rank,
                        non_blocking=True,
                    ),
                )
                y, y_lengths = (
                    y.cuda(
                        rank,
                        non_blocking=True,
                    ),
                    y_lengths.cuda(
                        rank,
                        non_blocking=True,
                    ),
                )
                ssl = ssl.cuda(rank, non_blocking=True)
                ssl.requires_grad = False
                # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
                text, text_lengths = (
                    text.cuda(
                        rank,
                        non_blocking=True,
                    ),
                    text_lengths.cuda(
                        rank,
                        non_blocking=True,
                    ),
                )
            else:
                spec, spec_lengths = spec.to(device), spec_lengths.to(device)
                y, y_lengths = y.to(device), y_lengths.to(device)
                ssl = ssl.to(device)
                ssl.requires_grad = False
                # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
                text, text_lengths = text.to(device), text_lengths.to(device)
            
            with autocast(enabled=hps.train.fp16_run):
                (
                    y_hat,
                    kl_ssl,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    stats_ssl,
                ) = net_g(ssl, spec, spec_lengths, text, text_lengths)
                
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r,
                        y_d_hat_g,
                    )
                    loss_disc_all = loss_disc
            
            # TPU v4-32에서 메모리 효율적인 역전파
            optim_d.zero_grad()
            if scaler is None:
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                xm.optimizer_step(optim_d)
            else:
                scaler.scale(loss_disc_all).backward()
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
                scaler.step(optim_d)
                
            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

            # TPU v4-32에서 메모리 효율적인 역전파
            optim_g.zero_grad()
            if scaler is None:
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                xm.optimizer_step(optim_g)
                tracker.add(hps.train.batch_size // 2)
                xm.add_step_closure(
                    _train_update, args=(device, epoch, global_step, len(train_loader), loss_gen, tracker, writer)
                )
                xm.mark_step()
            else:
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
                scaler.step(optim_g)
                scaler.update()
            
            if rank == 0:
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl]
                    logger.info(
                        "Train Epoch: {} [{:.0f}%]".format(
                            epoch,
                            100.0 * batch_idx / len(train_loader),
                        )
                    )
                    logger.info([x.item() for x in losses] + [global_step, lr])
                    if is_tpu_available():
                        xm.master_print(
                            "Train Epoch: {} [{:.0f}%]".format(
                                epoch,
                                100.0 * batch_idx / len(train_loader),
                                )
                        )
                        xm.master_print(
                            [x.item() for x in losses] + [global_step, lr]
                        )

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl_ssl": kl_ssl,
                            "loss/g/kl": loss_kl,
                        }
                    )

                    # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                    # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                    # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                    image_dict = None
                    try:  ###Some people installed the wrong version of matplotlib.
                        image_dict = {
                            "slice/mel_org": utils.plot_spectrogram_to_numpy(
                                y_mel[0].data.cpu().numpy(),
                            ),
                            "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                                y_hat_mel[0].data.cpu().numpy(),
                            ),
                            "all/mel": utils.plot_spectrogram_to_numpy(
                                mel[0].data.cpu().numpy(),
                            ),
                            "all/stats_ssl": utils.plot_spectrogram_to_numpy(
                                stats_ssl[0].data.cpu().numpy(),
                            ),
                        }
                    except:
                        pass
                    if image_dict:
                        utils.summarize(
                            writer=writer,
                            global_step=global_step,
                            images=image_dict,
                            scalars=scalar_dict,
                        )
                    else:
                        utils.summarize(
                            writer=writer,
                            global_step=global_step,
                            scalars=scalar_dict,
                        )
                    if is_tpu_available():
                        import torch_xla.debug.metrics_compare_utils as mcu
                        import torch_xla.debug.metrics as met
                        import torch_xla.core.xla_model as xm

                        metrics = mcu.parse_metrics_report(met.metrics_report())
                        aten_ops_sum = 0
                        for metric_name, metric_value in metrics.items():
                            if metric_name.find('aten::') == 0:
                                aten_ops_sum += metric_value
                            writer.add_scalar(metric_name, metric_value, global_step)
                            writer.add_scalar('aten_ops_sum', aten_ops_sum, global_step)
            global_step += 1
    except Exception as e:
        # TPU 메모리 오류 처리
        if is_tpu_available():
            import torch_xla.core.xla_model as xm
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                memory_error_count += 1
                xm.master_print(f"TPU 메모리 오류 발생 ({memory_error_count}/{max_memory_errors}): {str(e)}")
                
                if memory_error_count >= max_memory_errors and rank == 0:
                    xm.master_print("지속적인 TPU 메모리 오류로 인해 배치 크기를 줄이는 것을 고려하세요.")
                    xm.master_print("config.json 파일에서 'batch_size' 값을 절반으로 줄이고 다시 시도하세요.")
                    
                # TPU 코어 동기화 강제 수행
                xm.mark_step()
                sync_tpu_cores()
            else:
                import traceback
                import gc
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                
                # 기타 오류 처리
                xm.master_print(f"학습 중 오류 발생: {str(e)}")
        else:
            # 비-TPU 환경에서의 오류 처리
            logging.error(f"학습 중 오류 발생: {str(e)}")
    
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
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
        if rank == 0 and hps.train.if_save_every_weights == True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        global_step,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            text,
            text_lengths,
        ) in enumerate(eval_loader):
            if torch.cuda.is_available():
                spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
                y, y_lengths = y.cuda(), y_lengths.cuda()
                ssl = ssl.cuda()
                text, text_lengths = text.cuda(), text_lengths.cuda()
            else:
                spec, spec_lengths = spec.to(device), spec_lengths.to(device)
                y, y_lengths = y.to(device), y_lengths.to(device)
                ssl = ssl.to(device)
                text, text_lengths = text.to(device), text_lengths.to(device)
            for test in [0, 1]:
                y_hat, mask, *_ = (
                    generator.module.infer(
                        ssl,
                        spec,
                        spec_lengths,
                        text,
                        text_lengths,
                        test=test,
                    )
                    if torch.cuda.is_available()
                    else generator.infer(
                        ssl,
                        spec,
                        spec_lengths,
                        text,
                        text_lengths,
                        test=test,
                    )
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}_{test}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy(),
                        ),
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{test}": y_hat[0, :, : y_hat_lengths[0]],
                    },
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
                    },
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

        # y_hat, mask, *_ = generator.module.infer(ssl, spec_lengths, speakers, y=None)
        # audio_dict.update({
        #     f"gen/audio_{batch_idx}_style_pred": y_hat[0, :, :]
        # })

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    # start logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("train.log"),
            logging.StreamHandler(),
        ],  # 파일과 콘솔에 동시에 로깅
    )
        
    main()
