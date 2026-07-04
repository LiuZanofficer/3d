import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MaxMetric
from torchmetrics.classification.confusion_matrix import MulticlassConfusionMatrix

import src.utils.caption_utils as caption_utils
from src.models.lightning_modules.module_base import LitModuleBase
from src.models.losses.caption_loss import (
    CaptionAlignmentLoss,
    CaptionCLIPLoss,
    CaptionLoss,
    CaptionSigLIPLoss,
    DenseCaptionAlignmentLoss,
    HardNegativeCaptionLoss,
)
from src.models.losses.clip_alignment_loss import CLIPAlignmentEval
from src.models.utils.clip_models import build_clip_model, download_clip_model
from src.models.utils.evaluator import InstanceSegmentationEvaluator
from src.models.utils.structure import Point
from src.utils import RankedLogger
from src.utils.class_term_utils import build_text_clusters, class_to_cluster
from src.utils.readout_decorrelate import whiten_pick

log = RankedLogger(__file__, rank_zero_only=True)


class DenseLanguageLitModule(LitModuleBase):
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        scheduler_interval: str,
        clip_encoder: Dict,
        compile: bool,
        loss_cfg: Dict,
        best_metric: str,
        eval_cfg: Optional[Dict] = None,
        use_prompt: bool = False,
    ):
        super().__init__()

        # clip_encoder is frozen and rebuilt in configure_model; it is saved into
        # the checkpoint but is a non-registered object at load time. Allow
        # non-strict loading so resuming ignores those extra clip_encoder.* keys.
        self.strict_loading = False

        self.save_hyperparameters(logger=False)

        self.net = None

        # Mix3D augmentations
        self.mix_prob = loss_cfg.get("mix_prob", 0)

        # loss functions
        self.caption_loss_type = loss_cfg["caption_loss"].get("type", "contrastive")
        if self.caption_loss_type == "contrastive":
            self.caption_loss = CaptionLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "alignment":
            self.caption_loss = DenseCaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "region_alignment":
            self.caption_loss = CaptionAlignmentLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "clip":
            self.caption_loss = CaptionCLIPLoss(**loss_cfg["caption_loss"])
        elif self.caption_loss_type == "siglip":
            self.caption_loss = CaptionSigLIPLoss(**loss_cfg["caption_loss"])
        else:
            raise ValueError(f"Caption loss type {self.caption_loss_type} not supported")
        self.hard_negative_caption_loss = None
        if loss_cfg.get("hard_negative_caption_loss", None) is not None:
            self.hard_negative_caption_loss = HardNegativeCaptionLoss(
                **loss_cfg["hard_negative_caption_loss"]
            )

        # for tracking best so far validation accuracy
        self.val_metrics = nn.ModuleDict()
        self.val_class_info = dict()
        self.val_dataset_names = dict()
        self.val_best_metric = MaxMetric()

        # Save val_best_metric to hparams and restore if resuming
        self.save_hyperparameters({"val_best_metric": self.val_best_metric})

        # Sync distributed metrics
        self.train_sync_dist = loss_cfg.get("sync_dist", False)

        # eval configs
        self.ignore_background = False
        self.ignore_class_prob = False
        self.val_datasets = dict()
        self.val_loader_batch_sizes = dict()

    def prepare_data(self) -> None:
        # download clip model on rank 0
        ckpt_path = download_clip_model(self.hparams.clip_encoder)
        log.info(f"Downloaded CLIP model to {ckpt_path}")

    def configure_model(self) -> None:
        # network
        if self.net is not None:
            return

        self.net = self.hparams.net()
        # Print network on the first GPU
        if self.local_rank == 0:
            log.info(self.net)

        # clip encoder is frozen and used only for text encoding. Keep it out of
        # Lightning's registered submodules so DDP does not wrap/synchronize it.
        clip_encoder = build_clip_model(self.hparams.clip_encoder, device=self.device)
        for params in clip_encoder.parameters():
            params.requires_grad = False
        if "clip_encoder" in self._modules:
            del self._modules["clip_encoder"]
        object.__setattr__(self, "clip_encoder", clip_encoder)

    def on_load_checkpoint(self, checkpoint):
        if hasattr(self.hparams, "val_best_metric"):
            value = checkpoint["hyper_parameters"].get("val_best_metric", None)
            if value is not None:
                self.val_best_metric.update(value.max_value)
        super().on_load_checkpoint(checkpoint)

    def on_fit_start(self):
        init_ckpt_path = getattr(self, "init_ckpt_path", None)
        if init_ckpt_path and not getattr(self, "_init_ckpt_loaded", False):
            ckpt = torch.load(init_ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            for key in list(state_dict.keys()):
                if "emb_target" in key:
                    del state_dict[key]
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            self._init_ckpt_loaded = True
            log.info(
                f"Loaded init_ckpt_path={init_ckpt_path} "
                f"with missing={len(missing)} unexpected={len(unexpected)}"
            )

    def setup(self, stage: str) -> None:
        val_dataloaders = self.trainer.datamodule.val_dataloader()
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]

        for i, val_dataloader in enumerate(val_dataloaders):
            dataset = val_dataloader.dataset
            class_names = dataset.CLASS_LABELS
            postfix = dataset.log_postfix
            assert postfix is not None, "log_postfix is required for clarity"

            # semantic segmentation metrics (default)
            val_metric = nn.ModuleDict(
                {
                    "confmat": MulticlassConfusionMatrix(
                        num_classes=len(class_names),
                        ignore_index=dataset.ignore_label,
                    ),
                    "confmat_all": MulticlassConfusionMatrix(
                        num_classes=len(class_names),
                        ignore_index=dataset.ignore_label,
                    ),
                }
            )
            # instance segmentation metrics (optional)
            if dataset.mask_dir is not None:
                val_metric["mAP_evaluator"] = InstanceSegmentationEvaluator(
                    class_names=class_names,
                    segment_ignore_index=dataset.instance_ignore_class_idx
                    + [dataset.ignore_label],
                    instance_ignore_index=dataset.ignore_label,
                    subset_mapper=dataset.subset_mapper,
                )
            # dataset class info
            val_class_info = dict(
                postfix=postfix,
                class_names=class_names,
                base_class_idx=dataset.base_class_idx
                if hasattr(dataset, "base_class_idx")
                else None,
                novel_class_idx=dataset.novel_class_idx
                if hasattr(dataset, "novel_class_idx")
                else None,
                fg_class_idx=dataset.fg_class_idx if hasattr(dataset, "fg_class_idx") else None,
                bg_class_idx=dataset.bg_class_idx if hasattr(dataset, "bg_class_idx") else None,
                ignore_label=dataset.ignore_label,
                instance_ignore_class_idx=dataset.instance_ignore_class_idx
                if hasattr(dataset, "instance_ignore_class_idx")
                else None,
                subset_mapper=dataset.subset_mapper if hasattr(dataset, "subset_mapper") else None,
            )
            self.val_metrics[postfix] = val_metric
            self.val_class_info[postfix] = val_class_info
            self.val_dataset_names[i] = postfix
            self.val_datasets[i] = dataset
            self.val_loader_batch_sizes[i] = getattr(val_dataloader, "batch_size", None)

        self.clip_alignment_eval = nn.ModuleDict(
            {
                postfix: CLIPAlignmentEval(**self.hparams.eval_cfg.seg_eval)
                for postfix in self.val_metrics.keys()
            }
        )

    def forward(self, batch: Any) -> Dict[str, Any]:
        point = self.net(batch)
        out_dict = self._output_to_dict(point, batch)
        return out_dict

    def _output_to_dict(self, output: Any, batch: Any) -> Dict[str, Any]:
        assert isinstance(output, Point)
        output: Point = output
        clip_feat = output.sparse_conv_feat.features[output.v2p_map]
        out_dict = dict(point=output, clip_feat=clip_feat)
        return out_dict

    def training_step(self, batch, batch_idx):
        self._train_start = time.time()
        self.clip_encoder = self.clip_encoder.to(self.device)

        if random.random() < self.mix_prob:
            offset = batch["offset"]
            batch["offset"] = torch.cat([offset[1:-1:2], offset[-1].unsqueeze(0)], dim=0)

        # Time forward pass
        self._forward_start = time.time()
        out_dict = self(batch)
        clip_feat = out_dict["clip_feat"]
        forward_time = time.time() - self._forward_start
        self.forward_time(forward_time)

        # loss
        caption_loss = 0

        # Time loss computation
        self._loss_start = time.time()

        caption_loss_kargs = {
            "captions": batch["caption_data"].get("caption", None),
            "embeddings": batch["caption_data"].get("embedding", None),
            "point_indices": batch["caption_data"]["point_indices"],
            "caption_offsets": batch["caption_data"]["caption_offsets"],
            "num_points_per_caption": batch["caption_data"]["num_points_per_caption"],
            "clip_encoder": self.clip_encoder,
        }
        caption_loss = (
            self.caption_loss.loss(clip_feat, **caption_loss_kargs)
            * self.hparams.loss_cfg.weights.caption_loss
        )

        hard_negative_caption_loss = clip_feat.sum() * 0.0
        if self.hard_negative_caption_loss is not None:
            hard_negative_caption_loss = (
                self.hard_negative_caption_loss.loss(clip_feat, **caption_loss_kargs)
                * self.hparams.loss_cfg.weights.get("hard_negative_caption_loss", 0.0)
            )

        loss = caption_loss + hard_negative_caption_loss
        loss_time = time.time() - self._loss_start
        self.loss_time(loss_time)

        lr = self.optimizers().param_groups[0]["lr"]
        log_metrics = dict(
            loss=loss,
            caption_loss=caption_loss,
            hard_negative_caption_loss=hard_negative_caption_loss,
            lr=lr,
        )

        # useful metadata
        bs = len(batch["offset"]) - 1
        log_metrics["num_points"] = batch["coord"].shape[0] / bs
        log_metrics["num_objects"] = (batch["caption_data"]["caption_offsets"].shape[0] - 1) / bs

        # Calculate training time and mark start of next data loading
        train_time = time.time() - self._train_start
        self.train_time(train_time)
        self._data_load_start = time.time()

        # Add timing metrics to existing logging
        log_metrics.update(
            {
                "time/data_loading": self.data_load_time.compute(),
                "time/forward": self.forward_time.compute(),
                "time/loss": self.loss_time.compute(),
                "time/training": self.train_time.compute(),
            }
        )

        self.log_dict(
            {f"train/{key}": value for key, value in log_metrics.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=self.train_sync_dist,
        )
        return loss

    def on_validation_epoch_start(self):
        self.clip_encoder = self.clip_encoder.to(self.device)
        self._caption_proto_cache = {}
        for postfix in self.val_class_info.keys():
            class_info = self.val_class_info[postfix]
            eval_module = self.clip_alignment_eval[postfix]
            class_names = class_info["class_names"]

            anchor_npz = os.environ.get("MOSAIC3D_ANCHOR_NPZ", None)
            _injected = False
            if anchor_npz is not None:
                _a = np.load(anchor_npz, allow_pickle=True)
                _emb = torch.from_numpy(np.asarray(_a["emb_target"], dtype=np.float32))
                if _emb.shape[0] == len(class_names):
                    _emb = torch.nn.functional.normalize(_emb, dim=-1).to(self.device)
                    eval_module.set_target_embedding(_emb)
                    _injected = True
            if _injected:
                pass
            elif eval_module.emb_target is None:
                if self.hparams.use_prompt:
                    class_names = [
                        f"a {c} in a scene" if "other" not in c else "other" for c in class_names
                    ]  # OpenScene setting
                text_embedding = caption_utils.forward_text_encoder(
                    class_names, self.clip_encoder, normalize=True, device=self.device
                )
                eval_module.set_target_embedding(text_embedding.to(self.device))
            else:
                if eval_module.emb_target.device != self.device:
                    eval_module.emb_target = eval_module.emb_target.to(self.device)

            # reset metrics
            metrics = self.val_metrics[postfix]
            for key in metrics.keys():
                metrics[key].reset()

    def _readout_mode(self):
        return os.environ.get("MOSAIC3D_READOUT_MODE", "baseline")

    def _get_caption_proto_readout(self, postfix, class_info):
        if postfix in self._caption_proto_cache:
            return self._caption_proto_cache[postfix]

        proto_path = os.environ.get("MOSAIC3D_PROTO_PATH", None)
        if proto_path is None:
            raise RuntimeError("MOSAIC3D_PROTO_PATH is required for caption_proto readout")
        data = np.load(proto_path, allow_pickle=True)
        proto = torch.from_numpy(data["proto"]).float().to(self.device)
        proto_count = torch.from_numpy(data["count"]).long().to(self.device)
        proto = torch.nn.functional.normalize(proto, dim=-1)

        if self._readout_mode() == "gated_proto" and "gate_c2cluster" in data.files:
            import json as _json
            gmap = _json.loads(str(data["gate_c2cluster"]))
            c2cluster = {int(k): [int(x) for x in v] for k, v in gmap.items()}
            threshold = -1.0
        else:
            emb = self.clip_alignment_eval[postfix].emb_target.detach().float()
            emb_np = emb.cpu().numpy()
            threshold = float(os.environ.get("MOSAIC3D_CLUSTER_THRESHOLD", "0.88"))
            clusters = build_text_clusters(
                emb_np, np.array(class_info["fg_class_idx"], dtype=np.int64), threshold
            )
            c2cluster = class_to_cluster(clusters)
        cache = dict(proto=proto, proto_count=proto_count, c2cluster=c2cluster, threshold=threshold)
        self._caption_proto_cache[postfix] = cache
        return cache

    def _readout_sample(self, pred_logits, clip_feat, pred_masks, base_pred, class_info, postfix):
        mode = self._readout_mode()
        if mode == "baseline" or pred_masks is None or len(pred_masks) == 0:
            return base_pred, None, None

        num_classes = pred_logits.shape[-1]
        fg_idx = class_info["fg_class_idx"]
        logits_fg = torch.full_like(pred_logits, torch.finfo(pred_logits.dtype).min)
        logits_fg[..., fg_idx] = pred_logits[..., fg_idx]
        prob = torch.nn.functional.softmax(logits_fg, dim=-1)

        mask_probs = []
        masks = []
        for mask in pred_masks:
            mask = mask.to(device=pred_logits.device, dtype=torch.bool)
            if not torch.any(mask):
                continue
            masks.append(mask)
            mask_probs.append(prob[mask].mean(dim=0))
        if len(masks) == 0:
            return base_pred, None, None
        mask_probs = torch.stack(mask_probs, dim=0)
        pred_scores, pred_classes = torch.max(mask_probs, dim=1)

        if mode in ("caption_proto", "gated_proto"):
            readout = self._get_caption_proto_readout(postfix, class_info)
            proto = readout["proto"]
            proto_count = readout["proto_count"]
            c2cluster = readout["c2cluster"]
            feat = torch.nn.functional.normalize(clip_feat.detach().float(), dim=-1)
            new_classes = []
            for cls, mask in zip(pred_classes, masks):
                c0 = int(cls.item())
                cluster = c2cluster.get(c0, [c0])
                valid = [c for c in cluster if c < num_classes and int(proto_count[c].item()) > 0]
                if len(valid) <= 1:
                    new_classes.append(cls)
                    continue
                inst_feat = feat[mask].mean(dim=0)
                inst_feat = torch.nn.functional.normalize(inst_feat, dim=0)
                valid_t = torch.tensor(valid, dtype=torch.long, device=proto.device)
                sim = inst_feat @ proto[valid_t].T
                new_classes.append(valid_t[sim.argmax()])
            pred_classes = torch.stack(new_classes).to(pred_logits.device)
        elif mode == "anchor_decorrelate":
            emb = self.clip_alignment_eval[postfix].emb_target.detach().float().cpu().numpy()
            thr = float(os.environ.get("MOSAIC3D_CLUSTER_THRESHOLD", "0.90"))
            max_size = int(os.environ.get("MOSAIC3D_MAX_CLUSTER_SIZE", "8"))
            clusters = build_text_clusters(
                emb, np.array(class_info["fg_class_idx"], dtype=np.int64), thr
            )
            c2cluster = class_to_cluster(clusters)
            emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            feat = torch.nn.functional.normalize(clip_feat.detach().float(), dim=-1)
            new_classes = []
            for cls, mask in zip(pred_classes, masks):
                c0 = int(cls.item())
                members = c2cluster.get(c0, [c0])
                if not (2 <= len(members) <= max_size):
                    new_classes.append(cls)
                    continue
                inst_feat = torch.nn.functional.normalize(feat[mask].mean(dim=0), dim=0)
                j = whiten_pick(inst_feat.cpu().numpy(), emb_n[members])
                new_classes.append(torch.tensor(members[j], device=pred_logits.device))
            pred_classes = torch.stack(new_classes).to(pred_logits.device)
        elif mode != "mask_text_vote":
            return base_pred, None, None

        out_pred = base_pred.clone()
        assigned = torch.zeros_like(base_pred, dtype=torch.bool)
        for idx in torch.argsort(pred_scores, descending=True):
            mask = masks[int(idx.item())]
            write_mask = mask & (~assigned)
            if torch.any(write_mask):
                out_pred[write_mask] = pred_classes[idx]
                assigned[write_mask] = True
        return out_pred, pred_classes, pred_scores

    def _readout_batch(self, batch, logits, clip_feat, base_preds, class_info, postfix):
        if self._readout_mode() == "baseline" or "masks_binary" not in batch:
            return base_preds
        offset = batch["offset"]
        out = base_preds.clone()
        for i in range(len(offset) - 1):
            start, end = offset[i], offset[i + 1]
            sample_pred, _, _ = self._readout_sample(
                pred_logits=logits[start:end],
                clip_feat=clip_feat[start:end],
                pred_masks=batch["masks_binary"][i],
                base_pred=base_preds[start:end],
                class_info=class_info,
                postfix=postfix,
            )
            out[start:end] = sample_pred
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._current_batch_idx = batch_idx
        self._current_dataloader_idx = dataloader_idx
        postfix = self.val_dataset_names[dataloader_idx]
        metrics = self.val_metrics[postfix]
        class_info = self.val_class_info[postfix]

        out_dict = self(batch)
        logits = self.clip_alignment_eval[postfix].predict(
            out_dict["clip_feat"], return_logit=True
        )
        if os.environ.get("MOSAIC3D_DUMP_EVAL", "0") == "1" and self.trainer.is_global_zero:
            self._accumulate_visual_means(
                postfix, out_dict["clip_feat"], batch["segment"], class_info
            )
        if os.environ.get("SAVE_PRED", None) is not None:
            pred_save_dir = os.path.join(
                os.path.dirname(os.path.dirname(self.logger.log_dir)), "pred"
            )
            if not os.path.exists(pred_save_dir):
                os.makedirs(pred_save_dir, exist_ok=True)
            torch.save(
                {
                    "feat": out_dict["clip_feat"].cpu(),
                    "coord": batch["origin_coord"].cpu(),
                    "pc_count": batch["pc_count"].cpu(),
                },
                os.path.join(pred_save_dir, f"pred_{batch_idx}.pth"),
            )

        # 1. semantic segmentation
        preds_all = logits.max(1)[1]
        metrics["confmat_all"](preds_all, batch["segment"])

        logits_fg = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits_fg[..., class_info["fg_class_idx"]] = logits[..., class_info["fg_class_idx"]]

        preds = logits_fg.max(1)[1]
        preds_scored = self._readout_batch(
            batch=batch,
            logits=logits,
            clip_feat=out_dict["clip_feat"],
            base_preds=preds,
            class_info=class_info,
            postfix=postfix,
        )
        segment_fg = batch["segment"].clone()
        for i in class_info["bg_class_idx"]:
            segment_fg[segment_fg == i] = class_info["ignore_label"]  # Set background classes to 0

        # update and log metrics
        metrics["confmat"](preds_scored, segment_fg)

        # 2. instance segmentation (optional)
        if "mAP_evaluator" in metrics:
            # reuse `preds` (foreground-only argmax, same tensor used for f-mIoU)
            # so the dump's pred_semantic is exactly the scored prediction.
            self._update_instance_segmentation_metrics(
                batch,
                logits,
                metrics,
                class_info,
                pred_semantic_full=preds_scored,
                clip_feat_full=out_dict["clip_feat"],
            )

    def _update_instance_segmentation_metrics(
        self,
        batch,
        logits,
        metrics,
        class_info,
        pred_semantic_full=None,
        clip_feat_full=None,
    ):
        offset = batch["offset"]
        batch_size = len(offset) - 1
        ignore_class_idx = class_info["instance_ignore_class_idx"]
        for i in range(batch_size):
            gt_classes = batch["segment"][offset[i] : offset[i + 1]]
            gt_instances = batch["instance"][offset[i] : offset[i + 1]]
            pred_logits = logits[offset[i] : offset[i + 1]]
            pred_masks = batch["masks_binary"][i]

            # mask logits (voting)
            pred_logits_fg = pred_logits.clone()

            if self.ignore_background:
                pred_logits_fg[..., ignore_class_idx] = torch.finfo(pred_logits.dtype).min

            pred_logits_fg = torch.nn.functional.softmax(pred_logits_fg, dim=-1)
            pred_point_classes = pred_logits_fg.argmax(dim=1)
            pred_logits_fg = torch.stack([pred_logits_fg[mask].mean(dim=0) for mask in pred_masks])
            pred_scores, pred_classes = torch.max(pred_logits_fg, dim=1)

            if self.ignore_class_prob:
                pred_scores = torch.ones_like(pred_scores)

            # semantic-scored prediction (foreground-only argmax). Reuse the tensor
            # already computed in validation_step instead of reallocating a full
            # [N, num_classes] masked-logits tensor (saves memory on big scenes).
            pred_semantic = None
            if pred_semantic_full is not None:
                pred_semantic = pred_semantic_full[offset[i] : offset[i + 1]]
            clip_feat = None
            if clip_feat_full is not None:
                clip_feat = clip_feat_full[offset[i] : offset[i + 1]]
            if self._readout_mode() != "baseline" and clip_feat is not None:
                _, readout_classes, readout_scores = self._readout_sample(
                    pred_logits=pred_logits,
                    clip_feat=clip_feat,
                    pred_masks=pred_masks,
                    base_pred=pred_semantic if pred_semantic is not None else pred_point_classes,
                    class_info=class_info,
                    postfix=class_info["postfix"],
                )
                if readout_classes is not None and len(readout_classes) == len(pred_classes):
                    pred_classes = readout_classes.to(pred_classes.device)
                    pred_scores = readout_scores.to(pred_scores.device)

            self._dump_eval_scene_prediction(
                batch=batch,
                batch_idx=getattr(self, "_current_batch_idx", None),
                dataloader_idx=getattr(self, "_current_dataloader_idx", None),
                sample_idx=i,
                pred_point_classes=pred_point_classes,
                pred_semantic=pred_semantic,
                pred_mask_classes=pred_classes,
                pred_mask_scores=pred_scores,
                pred_masks=pred_masks,
                gt_segment=gt_classes,
                gt_instance=gt_instances,
                class_info=class_info,
                clip_feat=clip_feat,
            )

            metrics["mAP_evaluator"].update(
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                pred_masks=pred_masks,
                gt_segment=gt_classes,
                gt_instance=gt_instances,
            )

    def _dump_eval_scene_prediction(
        self,
        batch,
        batch_idx,
        dataloader_idx,
        sample_idx,
        pred_point_classes,
        pred_mask_classes,
        pred_mask_scores,
        pred_masks,
        gt_segment,
        gt_instance,
        class_info,
        pred_semantic=None,
        clip_feat=None,
    ):
        if os.environ.get("MOSAIC3D_DUMP_EVAL", "0") != "1":
            return
        if batch_idx is None or dataloader_idx is None:
            return
        if not self.trainer.is_global_zero:
            return

        dataset = self.val_datasets.get(dataloader_idx)
        if dataset is None or not hasattr(dataset, "scene_names"):
            return

        loader_batch_size = self.val_loader_batch_sizes.get(dataloader_idx) or (len(batch["offset"]) - 1)
        scene_idx = batch_idx * loader_batch_size + sample_idx
        if scene_idx >= len(dataset.scene_names):
            return
        scene_name = dataset.scene_names[scene_idx]

        output_root = Path(getattr(self.trainer, "default_root_dir", os.getcwd()))
        dump_dir = output_root / "eval_scene_dumps" / class_info["postfix"]
        dump_dir.mkdir(parents=True, exist_ok=True)

        save_kwargs = dict(
            scene_name=np.array(scene_name),
            class_names=np.array(class_info["class_names"], dtype=object),
            pred_point_classes=pred_point_classes.detach().cpu().numpy(),
            pred_mask_classes=pred_mask_classes.detach().cpu().numpy(),
            pred_mask_scores=pred_mask_scores.detach().cpu().numpy(),
            pred_masks=pred_masks.detach().cpu().numpy().astype(bool),
            gt_segment=gt_segment.detach().cpu().numpy(),
            gt_instance=gt_instance.detach().cpu().numpy(),
        )

        # semantic-scored prediction + class-index metadata, so downstream
        # analysis can reproduce the foreground f-mIoU exactly.
        if pred_semantic is not None:
            save_kwargs["pred_semantic"] = pred_semantic.detach().cpu().numpy()
        for key in ("fg_class_idx", "bg_class_idx", "instance_ignore_class_idx"):
            val = class_info.get(key, None)
            if val is not None:
                save_kwargs[key] = np.array(val, dtype=np.int64)
        save_kwargs["ignore_label"] = np.array(class_info.get("ignore_label", -100))

        np.savez_compressed(dump_dir / f"{scene_name}.npz", **save_kwargs)

        self._dump_eval_instance_features(
            output_root=output_root,
            postfix=class_info["postfix"],
            scene_name=scene_name,
            gt_segment=gt_segment,
            gt_instance=gt_instance,
            pred_semantic=pred_semantic,
            clip_feat=clip_feat,
            class_info=class_info,
        )

    def _dump_eval_instance_features(
        self,
        output_root,
        postfix,
        scene_name,
        gt_segment,
        gt_instance,
        pred_semantic,
        clip_feat,
        class_info,
    ):
        """Task 1: save one pooled visual feature per GT foreground instance.

        The feature matches the semantic decision space: normalize point features,
        average over the GT instance, then normalize the instance mean. Per-point
        features are intentionally not saved.
        """
        if pred_semantic is None or clip_feat is None:
            return

        num_classes = len(class_info["class_names"])
        ignore_label = class_info.get("ignore_label", -100)
        bg_idx = class_info.get("bg_class_idx", [])
        bg_mask = torch.zeros(num_classes, dtype=torch.bool, device=gt_segment.device)
        if len(bg_idx) > 0:
            bg_mask[torch.as_tensor(bg_idx, device=gt_segment.device).long()] = True

        feat = torch.nn.functional.normalize(clip_feat.detach().float(), dim=-1)
        gt = gt_segment.detach().long()
        inst = gt_instance.detach().long()
        pred = pred_semantic.detach().long()

        instance_ids = []
        true_classes = []
        n_points = []
        pooled_feats = []
        pred_majority_classes = []
        pred_majority_fracs = []
        pred_hists = []

        for inst_id in torch.unique(inst):
            inst_id_int = int(inst_id.item())
            if inst_id_int < 0:
                continue
            mask = inst == inst_id
            if not torch.any(mask):
                continue

            gt_vals = gt[mask]
            valid_gt = (gt_vals >= 0) & (gt_vals < num_classes) & (gt_vals != ignore_label)
            if not torch.any(valid_gt):
                continue
            gt_hist = torch.bincount(gt_vals[valid_gt], minlength=num_classes)
            true_class = int(torch.argmax(gt_hist).item())
            if bg_mask[true_class]:
                continue

            pred_vals = pred[mask]
            valid_pred = (pred_vals >= 0) & (pred_vals < num_classes)
            if torch.any(valid_pred):
                pred_hist = torch.bincount(pred_vals[valid_pred], minlength=num_classes)
            else:
                pred_hist = torch.zeros(num_classes, dtype=torch.long, device=pred.device)
            pred_majority = int(torch.argmax(pred_hist).item())
            total = int(mask.sum().item())
            pred_majority_frac = float(pred_hist[pred_majority].item() / max(total, 1))

            pooled = feat[mask].mean(dim=0)
            pooled = torch.nn.functional.normalize(pooled, dim=0)

            instance_ids.append(inst_id_int)
            true_classes.append(true_class)
            n_points.append(total)
            pooled_feats.append(pooled.cpu())
            pred_majority_classes.append(pred_majority)
            pred_majority_fracs.append(pred_majority_frac)
            pred_hists.append(pred_hist.cpu())

        if len(instance_ids) == 0:
            return

        out_dir = output_root / "eval_instance_features" / postfix
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"{scene_name}.npz",
            scene_name=np.array(scene_name),
            class_names=np.array(class_info["class_names"], dtype=object),
            gt_instance_id=np.array(instance_ids, dtype=np.int64),
            true_class=np.array(true_classes, dtype=np.int64),
            n_points=np.array(n_points, dtype=np.int64),
            pooled_feat=torch.stack(pooled_feats).numpy().astype(np.float32),
            pred_majority_class=np.array(pred_majority_classes, dtype=np.int64),
            pred_majority_frac=np.array(pred_majority_fracs, dtype=np.float32),
            pred_hist=torch.stack(pred_hists).numpy().astype(np.int64),
            fg_class_idx=np.array(class_info.get("fg_class_idx", []), dtype=np.int64),
            bg_class_idx=np.array(class_info.get("bg_class_idx", []), dtype=np.int64),
            ignore_label=np.array(ignore_label, dtype=np.int64),
        )

    def _accumulate_visual_means(self, postfix, clip_feat, gt_segment, class_info):
        """V1b: accumulate per-class sum of L2-normalized visual features (the same
        normalized features used for the argmax decision) so we can later compare
        each class's mean visual direction against the text embeddings."""
        feat = torch.nn.functional.normalize(clip_feat.detach().float(), dim=-1)
        gt = gt_segment.detach().to(feat.device).long()
        num_classes = len(class_info["class_names"])
        if not hasattr(self, "_vis_sum"):
            self._vis_sum, self._vis_cnt = {}, {}
        if postfix not in self._vis_sum:
            self._vis_sum[postfix] = torch.zeros(num_classes, feat.shape[1], device=feat.device)
            self._vis_cnt[postfix] = torch.zeros(num_classes, device=feat.device)
        valid = (gt >= 0) & (gt < num_classes)
        self._vis_sum[postfix].index_add_(0, gt[valid], feat[valid])
        self._vis_cnt[postfix].index_add_(
            0, gt[valid], torch.ones(int(valid.sum()), device=feat.device)
        )

    def _dump_visual_means(self, postfix, class_info):
        if os.environ.get("MOSAIC3D_DUMP_EVAL", "0") != "1":
            return
        if not getattr(self, "trainer", None) or not self.trainer.is_global_zero:
            return
        if not hasattr(self, "_vis_sum") or postfix not in self._vis_sum:
            return
        emb_target = self.clip_alignment_eval[postfix].emb_target
        out_root = Path(getattr(self.trainer, "default_root_dir", os.getcwd())) / "eval_visual_means"
        out_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_root / f"{postfix}.npz",
            class_names=np.array(class_info["class_names"], dtype=object),
            vis_sum=self._vis_sum[postfix].detach().cpu().numpy(),
            vis_cnt=self._vis_cnt[postfix].detach().cpu().numpy(),
            emb_target=emb_target.detach().cpu().numpy() if emb_target is not None else np.zeros(0),
            fg_class_idx=np.array(class_info.get("fg_class_idx", []), dtype=np.int64),
            bg_class_idx=np.array(class_info.get("bg_class_idx", []), dtype=np.int64),
        )

    def on_validation_epoch_end(self) -> None:
        def compute_classwise_metrics(confmat, class_names):
            computed_confmat = confmat.compute().cpu().numpy()
            class_ious = {}
            class_accs = {}
            for i, class_name in enumerate(class_names):
                tp = computed_confmat[i, i]
                fp = computed_confmat[:, i].sum() - tp
                fn = computed_confmat[i, :].sum() - tp

                class_ious[class_name] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                class_accs[class_name] = tp / (tp + fn) if (tp + fn) > 0 else 0

            return class_ious, class_accs

        log_metrics = {}
        for postfix, metrics in self.val_metrics.items():
            val_section = f"val_{postfix}"
            class_info = self.val_class_info[postfix]
            class_names = class_info["class_names"]

            # 1. semantic segmentation
            class_ious, class_accs = compute_classwise_metrics(metrics["confmat"], class_names)
            class_ious_all, class_accs_all = compute_classwise_metrics(
                metrics["confmat_all"], class_names
            )

            miou = np.nanmean([class_ious[class_names[i]] for i in class_info["fg_class_idx"]])
            macc = np.nanmean([class_accs[class_names[i]] for i in class_info["fg_class_idx"]])
            miou_all = np.nanmean([class_ious_all[c] for c in class_names])
            macc_all = np.nanmean([class_accs_all[c] for c in class_names])

            log_metrics.update({f"{val_section}/iou_{k}": v for k, v in class_ious_all.items()})
            log_metrics.update(
                {
                    f"{val_section}/miou": miou,
                    f"{val_section}/macc": macc,
                    f"{val_section}/miou_all": miou_all,
                    f"{val_section}/macc_all": macc_all,
                }
            )
            if class_info["subset_mapper"] is not None:
                subset_mapper = class_info["subset_mapper"]
                subset_names = subset_mapper["subset_names"]
                subset_mious = {}
                subset_maccs = {}
                for subset_name in subset_names:
                    subset_mious[subset_name] = np.nanmean(
                        [
                            class_ious[class_name]
                            for class_name in class_names
                            if subset_mapper[class_name] == subset_name
                        ]
                    )
                    subset_maccs[subset_name] = np.nanmean(
                        [
                            class_accs[class_name]
                            for class_name in class_names
                            if subset_mapper[class_name] == subset_name
                        ]
                    )
                log_metrics.update(
                    {
                        f"{val_section}/miou_{subset_name}": v
                        for subset_name, v in subset_mious.items()
                    }
                )
                log_metrics.update(
                    {
                        f"{val_section}/macc_{subset_name}": v
                        for subset_name, v in subset_maccs.items()
                    }
                )

            # V1b: dump per-class visual-feature means + text classifier
            self._dump_visual_means(postfix, class_info)

            # 2. instance segmentation (optional)
            if "mAP_evaluator" in metrics:
                instance_metrics = metrics["mAP_evaluator"].compute()
                classwise_aps = {}
                for class_name, classwise_metrics in instance_metrics["classes"].items():
                    for metric_name, metric_value in classwise_metrics.items():
                        classwise_aps[f"{val_section}/{metric_name}_{class_name}"] = metric_value
                log_metrics.update(classwise_aps)
                instance_metrics.pop("classes")
                log_metrics.update({f"{val_section}/{k}": v for k, v in instance_metrics.items()})

        # update best metric
        self.val_best_metric.update(log_metrics[self.hparams.best_metric])
        log_metrics.update({f"{self.hparams.best_metric}_best": self.val_best_metric.compute()})

        # log metrics only if not sanity checking
        if not self.trainer.sanity_checking:
            self.log_dict(log_metrics, sync_dist=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self._current_batch_idx = batch_idx
        self._current_dataloader_idx = dataloader_idx
        self.validation_step(batch, batch_idx, dataloader_idx)

    def children(self):
        for name, module in self.named_children():
            if name != "clip_encoder":
                yield module

    def parameters(self):
        for name, params in self.named_parameters():
            if "clip_encoder" not in name:
                yield params
