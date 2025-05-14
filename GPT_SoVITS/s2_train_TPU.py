import warnings

warnings.filterwarnings("ignore")
import os
import math
import utils

hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "0" 
os.environ["PT_XLA_DEBUG_LEVEL"] = "0"

# TPU 지원 추가
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPT_SoVITS.utils_tpu import is_tpu_available
import logging

# TPUv4 최적화 설정
TPU_OPTIMIZED_KWARGS = {
    'persistent_workers': True, # worker 가 데이터 로더를 계속 유지할지 여부
    'prefetch_factor': 1, # worker 가 미리 로드할 배치 수 (prefetch_factor * num_workers)
    'num_workers': 4, # 데이터 로더 worker 수

    'device_prefetch_size': 1, # TPU 에서 준비할 배치 수
    'loader_prefetch_size': 4, # CPU 에서 TPU에 보낼걸 미리 준비할 배치 수
    'host_to_device_transfer_threads': 4, # TPU에서 데이터 로더에서 호스트로 데이터를 전송하는 스레드 수
}

import torch
import torch.distributed as dist

from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
###反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
# from config import pretrained_s2G,pretrained_s2D
global_step = 0
# TPU 환경 변수 및 BF16 사용 권장


# 디바이스 설정

device = "cpu"  # cuda以外的设备，等mps优化后加入
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.metrics_compare_utils as mcu
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
from torch_xla import runtime as xr
from torch_xla.amp import syncfree, GradScaler, autocast


def run(index, hps):
    global global_step
    device = xm.xla_device()
    server = xp.start_server(9012)
    n_gpus = xr.world_size()
    rank = xr.global_ordinal()
    if rank == 0:
        os.environ["PT_XLA_DEBUG_LEVEL"] = "2"
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.s2_ckpt_dir)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    

    dist.init_process_group('xla', init_method='xla://', rank=rank, world_size=n_gpus)
    print(f"XLA:OPENED {rank}/{n_gpus}")

    train_dataset = TextAudioSpeakerLoader(hps.data, device_str=f"{device}:{rank}({n_gpus})")  ########

    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [ 32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900 ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    collate_fn = TextAudioSpeakerCollate()
    # 데이터 로더 생성 - TPU에 최적화된 설정
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        shuffle=False,
        num_workers=TPU_OPTIMIZED_KWARGS['num_workers'], # 동적으로 설정된 num_workers 사용
        persistent_workers=True, # TPU 환경에서는 True 권장
        prefetch_factor=TPU_OPTIMIZED_KWARGS['prefetch_factor'], # 기존 값 유지 또는 조정
        pin_memory=False
    )
    train_loader = pl.MpDeviceLoader(
        train_loader, 
        device,
        loader_prefetch_size=TPU_OPTIMIZED_KWARGS['loader_prefetch_size'],
        device_prefetch_size=TPU_OPTIMIZED_KWARGS['device_prefetch_size'],
        host_to_device_transfer_threads=TPU_OPTIMIZED_KWARGS['host_to_device_transfer_threads']
    )

    net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    
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

   
    optim_g = syncfree.AdamW(
        # filter(lambda p: p.requires_grad, net_g.parameters()),###默认所有层lr一致
        [
            {"params": base_params, "lr": hps.train.learning_rate},
            {
                "params": net_g.enc_p.text_embedding.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate * n_gpus,
            },
            {
                "params": net_g.enc_p.encoder_text.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate * n_gpus,
            },
            {
                "params": net_g.enc_p.mrte.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate * n_gpus,
            },
        ],
        hps.train.learning_rate * n_gpus,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = syncfree.AdamW(
        net_d.parameters(),
        hps.train.learning_rate * n_gpus,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # net_g = DDP(net_g, broadcast_buffers=False, gradient_as_bucket_view=True).to(device)
    # net_d = DDP(net_d, broadcast_buffers=False, gradient_as_bucket_view=True).to(device)

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
        epoch_str = 1
        global_step = 0
        try:
            if hps.train.pretrained_s2G != "" and hps.train.pretrained_s2G != None and os.path.exists(hps.train.pretrained_s2G):
                print("load pretrained s2G", hps.train.pretrained_s2G)
                if hasattr(net_g, "module"):
                    net_g.module.load_state_dict(
                        torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                        strict=False,
                    )
                else:
                    net_g.load_state_dict(
                        torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                        strict=False,
                    )
                
            if hps.train.pretrained_s2D != "" and hps.train.pretrained_s2D != None and os.path.exists(hps.train.pretrained_s2D):
                print("load pretrained s2D", hps.train.pretrained_s2D)
                if hasattr(net_d, "module"):
                    net_d.module.load_state_dict(
                        torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"],
                    )
                else:
                    net_d.load_state_dict(
                        torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"],
                    )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception("Failed to load pre-trained model") from e



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

    scaler = GradScaler(use_zero_grad=True, enabled=hps.train.fp16_run)
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
            
    xm.master_print("학습 완료")

def _get_device_spec(device):
    ordinal = xr.global_ordinal()
    return str(device) if ordinal < 0 else '{}/{}'.format(device, ordinal)

def _debug_print(device, message):
    print(f"[{_get_device_spec(device)}] {message}")    

def _train_update(device, epoch, step, total_step, loss, tracker):
    rate = tracker.rate()
    global_rate = tracker.global_rate()
    print(
        f"[Train] {_get_device_spec(device)} Epoch: [{epoch}/{hps.train.epochs}][{step}/{total_step}] "
        f"Loss: {loss.item():.4f} Rate: {rate:.4f} Global Rate: {global_rate:.4f} ",
        flush=True,
    )

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    global global_step
    tracker = xm.RateTracker()
    device = xm.xla_device()

    xm.master_print(f"에포크 {epoch}: 학습 시작")

    with autocast(device=device, enabled=hps.train.fp16_run):
        net_g, net_d = nets[0].to(device), nets[1].to(device)
        optim_g, optim_d = optims
        scheduler_g, scheduler_d = schedulers
        train_loader, eval_loader = loaders
        if writers is not None:
            writer, writer_eval = writers
        else:
            writer, writer_eval = None, None

        net_g.train()
        net_d.train()

    for batch_idx, ( ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, ) in enumerate(train_loader):
        xm.add_step_closure( _debug_print, args=(device, f"move_device") )
        with autocast(device=device, enabled=hps.train.fp16_run):
            spec, spec_lengths = spec.to(device, torch.float32), spec_lengths.to(device)
            y, y_lengths = y.to(device, torch.float32), y_lengths.to(device)
            text, text_lengths = text.to(device), text_lengths.to(device)
            ssl = ssl.to(device, torch.float32)
            ssl.requires_grad = False
            xm.add_step_closure( _debug_print, args=(device, f"forward") )
            # Move ssl to device just before use inside autocast
            xm.mark_step()
            assert not torch.is_complex(ssl), f"ssl is complex! dtype: {ssl.dtype} : info {ssl}"
            (
                y_hat,
                kl_ssl,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                stats_ssl,
            ) = net_g(ssl, spec, spec_lengths, text, text_lengths)
            # print(f"ssl dtype: {ssl.dtype}, spec dtype: {spec.dtype}, spec_lengths: {spec_lengths}, text dtype: {text.dtype}, text_lengths: {text_lengths}, y dtype: {y.dtype}, y_lengths: {y_lengths}")
            assert not torch.is_complex(y_hat), f"y_hat is complex! dtype: {y_hat.dtype} : info {y_hat}"
            xm.add_step_closure( _debug_print, args=(device, f"forward done") )
            xm.mark_step()
            spec = spec.float()
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            assert not torch.is_complex(mel), f"mel is complex! dtype: {mel.dtype} : info {mel}"
            xm.add_step_closure( _debug_print, args=(device, f"mel done") )
            xm.mark_step()
            mel = mel.float()
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            assert not torch.is_complex(y_mel), f"y_mel is complex! dtype: {y_mel.dtype} : info {y_mel}"
            # 항상 실수 텐서로 변환하여 일관성 유지
            # y_mel = y_mel.to(torch.float32)
            xm.mark_step()
            # Ensure we're working with real tensors before mel spectrogram calculation
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
            assert not torch.is_complex(y_hat_mel), f"y_hat_mel is complex! dtype: {y_hat_mel.dtype} : info {y_hat_mel}"
            # 항상 실수 텐서로 변환
            # y_hat_mel = y_hat_mel.to(torch.float32)
            xm.add_step_closure( _debug_print, args=(device, f"y_mel done") )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
            xm.mark_step()
            assert not torch.is_complex(y), f"y is complex! dtype: {y.dtype} : info {y}"
            y_hat_detach = y_hat.detach()
            y_hat_detach = y_hat_detach.float()
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat_detach)
            
            xm.add_step_closure( _debug_print, args=(device, f"y_d_hat done") )
            xm.mark_step()
            with autocast(device=device, enabled=False):
                
                loss_disc_all, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r,
                    y_d_hat_g,
                )
                assert not torch.is_complex(loss_disc_all), f"loss_disc is complex! dtype: {loss_disc_all.dtype} : info {loss_disc_all}"
                xm.add_step_closure( _debug_print, args=(device, f"loss_disc done") )
        xm.mark_step()
        xm.add_step_closure( _debug_print, args=(device, f"backward") )
        optim_d.zero_grad()
        scaler.scale(loss_disc_all.float()).backward()
        gradients = xm._fetch_gradients(optim_d)
        xm.all_reduce(xm.REDUCE_SUM, gradients, scale=1.0 / xr.world_size(), pin_layout=False)
        scaler.unscale_(optim_d)
        scaler.step(optim_d)
        xm.add_step_closure( _debug_print, args=(device, f"backward done") )
        with autocast(device=device, enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat.float())
            xm.mark_step()
            with autocast(device=device, enabled=False):
                loss_mel = F.l1_loss(y_mel.float(), y_hat_mel.float()) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                xm.mark_step()
                loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
        
        xm.add_step_closure( _debug_print, args=(device, f"loss_gen done") )
        optim_g.zero_grad()
        scaler.scale(loss_gen_all.float()).backward()
        gradients = xm._fetch_gradients(optim_g)
        xm.all_reduce(xm.REDUCE_SUM, gradients, scale=1.0 / xr.world_size(), pin_layout=False)
        scaler.unscale_(optim_g)
        scaler.step(optim_g)
        scaler.update()
        xm.add_step_closure( _debug_print, args=(device, f"backward done") )

        xm.add_step_closure(
            _train_update,
            args=(device, epoch, batch_idx, len(train_loader), loss_gen_all, tracker),
        )
        scheduler_g.step()
        scheduler_d.step()

        if rank == 0 and writer is not None:
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
                
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl_ssl": kl_ssl,
                        "loss/g/kl": loss_kl,
                    }
                )

                utils.summarize( writer=writer, global_step=global_step, scalars=scalar_dict, )

                metrics = mcu.parse_metrics_report(met.metrics_report())
                aten_ops_sum = 0
                for metric_name, metric_value in metrics.items():
                    if metric_name.find('aten::') == 0:
                        aten_ops_sum += metric_value
                    writer.add_scalar(metric_name, metric_value, global_step)
                    writer.add_scalar('aten_ops_sum', aten_ops_sum, global_step)
        global_step += 1   
        xm.mark_step()


    def _save_checkpoint():
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

    # To Save the last checkpoint each VM
    if xm.is_master_ordinal():
        _save_checkpoint()
        if rank == 0:
            if hps.train.if_save_every_weights == True:
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
    if is_tpu_available():
        print(f"TPU 멀티프로세싱 시작")
        torch_xla.experimental.eager_mode(True)
        xr.initialize_cache('/tmp/cache', False)
        debug_single_process = False
        torch_xla.launch(
            run, args=(hps, ), debug_single_process=debug_single_process)
            


    