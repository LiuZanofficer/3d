from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.evaluation.tta_evaluator import TTAEvaluator
from src.models.losses.hpza_loss import HPZALoss
from src.models.losses.instance_loss import InstanceDiscriminativeLoss
from src.models.losses.pmtl_scheduler import PMTLScheduler
from src.models.losses.seg_loss import SegLoss
from src.models.losses.tdcl_loss import TDCLLoss
from src.models.networks.vlm.text_encoder import HashTextEncoder, build_text_encoder
from src.models.utils.metrics import compute_iou_from_conf, compute_subset_miou, update_confusion_matrix
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LZLitModule(L.LightningModule):
    def __init__(
        self,
        net,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig],
        scheduler_interval: str,
        loss_cfg: Optional[Dict] = None,
        loss: Optional[Dict] = None,
        eval_cfg: Optional[Dict] = None,
        text_encoder_cfg: Optional[Dict] = None,
        best_metric: str = "val/miou",
        zero_shot: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if not isinstance(net, nn.Module):
            net = hydra.utils.instantiate(net)
        self.net = net
        if loss_cfg is None:
            loss_cfg = loss or {}
        seg_cfg = loss_cfg.get("seg_loss", {})
        tdcl_cfg = loss_cfg.get("tdcl", None)
        if tdcl_cfg is not None:
            tdcl_dict = dict(tdcl_cfg) if not isinstance(tdcl_cfg, dict) else tdcl_cfg
            self.seg_loss = TDCLLoss(
                num_classes=int(tdcl_dict.get("num_classes", 200)),
                tail_class_indices=list(tdcl_dict.get("tail_class_indices", [])),
                ignore_label=loss_cfg.get("ignore_label", -100),
                ce_weight=float(tdcl_dict.get("ce_weight", seg_cfg.get("ce_weight", 1.0))),
                label_smoothing=float(
                    tdcl_dict.get("label_smoothing", seg_cfg.get("label_smoothing", 0.0))
                ),
                class_freq_path=str(
                    tdcl_dict.get("class_freq_path", seg_cfg.get("class_freq_path", ""))
                ),
                class_weight_clip_min=float(tdcl_dict.get("class_weight_clip_min", 0.2)),
                class_weight_clip_max=float(tdcl_dict.get("class_weight_clip_max", 5.0)),
                tfr_weight=float(tdcl_dict.get("tfr_weight", 0.5)),
                ps_weight=float(tdcl_dict.get("ps_weight", 0.2)),
                ps_margin=float(tdcl_dict.get("ps_margin", 0.3)),
                ps_top_k=int(tdcl_dict.get("ps_top_k", 5)),
                ps_temperature=float(tdcl_dict.get("ps_temperature", 0.1)),
                gbs_weight=float(tdcl_dict.get("gbs_weight", 1.0)),
            )
            self._tdcl_active = True
        else:
            self.seg_loss = SegLoss(
                ignore_label=loss_cfg.get("ignore_label", -100),
                ce_weight=float(seg_cfg.get("ce_weight", 1.0)),
                lovasz_weight=float(seg_cfg.get("lovasz_weight", 0.0)),
                label_smoothing=float(seg_cfg.get("label_smoothing", 0.0)),
                use_class_weight=bool(seg_cfg.get("use_class_weight", False)),
                class_freq_path=str(seg_cfg.get("class_freq_path", "")),
                class_weight_mode=str(seg_cfg.get("class_weight_mode", "inv_sqrt")),
                class_weight_clip_min=float(seg_cfg.get("class_weight_clip_min", 0.2)),
                class_weight_clip_max=float(seg_cfg.get("class_weight_clip_max", 5.0)),
            )
            self._tdcl_active = False
        # Streaming train-confusion accumulator for the TDCL head-confuser table.
        self._train_conf: Optional[torch.Tensor] = None
        self.inst_loss = InstanceDiscriminativeLoss(
            ignore_label=loss_cfg.get("ignore_label", -100),
            delta_var=loss_cfg.get("instance_loss", {}).get("delta_var", 0.5),
            delta_dist=loss_cfg.get("instance_loss", {}).get("delta_dist", 1.5),
        )
        self.hpza_loss = HPZALoss(
            temperature=loss_cfg.get("hpza_loss", {}).get("temperature", 0.07),
            weight_caption=loss_cfg.get("hpza_loss", {}).get("weight_caption", 1.0),
            weight_class=loss_cfg.get("hpza_loss", {}).get("weight_class", 1.0),
        )
        self.pmtl = None
        if loss_cfg.get("pmtl") is not None:
            self.pmtl = PMTLScheduler(**loss_cfg["pmtl"])

        self.eval_cfg = eval_cfg or {}
        self.best_metric = best_metric
        self.zero_shot = zero_shot
        self.text_encoder_cfg = text_encoder_cfg or {"use_clip": False, "embed_dim": 768}
        self.text_encoder = None
        self.class_text_embeds = None
        self.class_names = None

        self.val_conf = {}
        self.val_class_names = {}
        self.val_subset_mapper = {}
        self.val_ignore_label = {}
        self.val_fg_class_idx = {}
        self.val_datasets = {}
        self._scene_ptr = {}
        self._input_stats_logged = False
        self._best_val_metric = float("-inf")
        self._best_val_diag = None
        self._last_val_diag = None
        self.eval_bn_batch_stats = bool(self.eval_cfg.get("eval_bn_batch_stats", False))
        self._eval_bn_state_cache: List[Dict[str, Any]] = []

        self.tta_evaluator = None
        if self.eval_cfg.get("use_tta", False):
            self.tta_evaluator = TTAEvaluator(
                rotations=self.eval_cfg.get("rotations", [0, 90, 180, 270]),
                flips=self.eval_cfg.get("flips", [False, True]),
                temperature=self.eval_cfg.get("temperature", 1.0),
                use_adapter_tta=self.eval_cfg.get("use_adapter_tta", True),
                adapter_steps=self.eval_cfg.get("adapter_steps", 1),
                adapter_lr=self.eval_cfg.get("adapter_lr", 1e-3),
                adapter_weight_decay=self.eval_cfg.get("adapter_weight_decay", 0.0),
                entropy_weight=self.eval_cfg.get("entropy_weight", 1.0),
            )

    def _batch_size_from_batch(self, batch: Dict[str, Any]) -> int:
        offsets = batch.get("offset")
        if offsets is None:
            return 1
        if isinstance(offsets, torch.Tensor):
            return max(int(offsets.numel()) - 1, 1)
        return max(len(offsets) - 1, 1)

    def _hpza_loss_weight(self) -> float:
        loss_cfg = self.hparams.get("loss_cfg")
        if loss_cfg is None:
            loss_cfg = self.hparams.get("loss")
        if loss_cfg is None:
            return 1.0
        weights = loss_cfg.get("weights", {})
        return float(weights.get("hpza_loss", 1.0))

    def _prediction_distribution_stats(
        self,
        preds: torch.Tensor,
        labels: Optional[torch.Tensor],
        ignore_label: int,
    ) -> Dict[str, float]:
        preds_flat = preds.view(-1).long()
        total_cnt = int(preds_flat.numel())
        if total_cnt == 0:
            return {
                "pred_unique_count": 0.0,
                "pred_top1_ratio": 0.0,
                "ignore_ratio": 0.0,
            }

        if labels is None:
            valid_mask = torch.ones_like(preds_flat, dtype=torch.bool)
        else:
            labels_flat = labels.view(-1).long()
            valid_mask = labels_flat != int(ignore_label)

        valid_preds = preds_flat[valid_mask]
        valid_cnt = int(valid_preds.numel())
        ignore_ratio = 1.0 - (valid_cnt / max(total_cnt, 1))
        if valid_cnt == 0:
            return {
                "pred_unique_count": 0.0,
                "pred_top1_ratio": 0.0,
                "ignore_ratio": float(ignore_ratio),
            }

        unique_count = float(torch.unique(valid_preds).numel())
        max_pred_id = int(valid_preds.max().item())
        counts = torch.bincount(valid_preds, minlength=max_pred_id + 1)
        top1_ratio = float(counts.max().item() / max(valid_cnt, 1))

        return {
            "pred_unique_count": unique_count,
            "pred_top1_ratio": top1_ratio,
            "ignore_ratio": float(ignore_ratio),
        }

    def _resolve_output_dir(self) -> Path:
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            loggers = getattr(trainer, "loggers", None)
            if loggers:
                for logger in loggers:
                    save_dir = getattr(logger, "save_dir", None)
                    if save_dir:
                        save_path = Path(str(save_dir))
                        if save_path.name == "logs":
                            return save_path.parent
                        return save_path
            logger = getattr(trainer, "logger", None)
            if logger is not None:
                save_dir = getattr(logger, "save_dir", None)
                if save_dir:
                    save_path = Path(str(save_dir))
                    if save_path.name == "logs":
                        return save_path.parent
                    return save_path
            default_root = getattr(trainer, "default_root_dir", None)
            if default_root:
                return Path(str(default_root))
        return Path.cwd()

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _set_eval_bn_batch_stats_mode(self) -> None:
        if not self.eval_bn_batch_stats:
            return
        if self._eval_bn_state_cache:
            self._restore_eval_bn_batch_stats_mode()

        self.eval()
        bn_states: List[Dict[str, Any]] = []
        for module in self.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            state: Dict[str, Any] = {
                "module": module,
                "training": bool(module.training),
                "momentum": module.momentum,
            }
            if hasattr(module, "num_batches_tracked") and isinstance(
                module.num_batches_tracked, torch.Tensor
            ):
                state["num_batches_tracked"] = module.num_batches_tracked.detach().clone()
            bn_states.append(state)
            module.train()
            module.momentum = 0.0
        self._eval_bn_state_cache = bn_states

    def _restore_eval_bn_batch_stats_mode(self) -> None:
        if not self._eval_bn_state_cache:
            return
        for state in self._eval_bn_state_cache:
            module = state["module"]
            module.train(mode=bool(state["training"]))
            module.momentum = state["momentum"]
            cached_nbt = state.get("num_batches_tracked")
            if (
                cached_nbt is not None
                and hasattr(module, "num_batches_tracked")
                and isinstance(module.num_batches_tracked, torch.Tensor)
            ):
                module.num_batches_tracked.copy_(
                    cached_nbt.to(device=module.num_batches_tracked.device)
                )
        self._eval_bn_state_cache = []

    def setup(self, stage: str) -> None:
        # build text encoder and class prototypes
        try:
            self.text_encoder = build_text_encoder(self.text_encoder_cfg)
        except Exception:
            log.warning("Falling back to HashTextEncoder.")
            self.text_encoder = HashTextEncoder(
                embed_dim=int(self.text_encoder_cfg.get("embed_dim", 768))
            )

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and datamodule.data_train is not None:
            datasets = []
            if hasattr(datamodule.data_train, "datasets"):
                datasets = datamodule.data_train.datasets
            else:
                datasets = [datamodule.data_train]
            # pick dataset with most classes
            datasets = [d for d in datasets if hasattr(d, "CLASS_LABELS")]
            if datasets:
                datasets = sorted(datasets, key=lambda d: len(d.CLASS_LABELS), reverse=True)
                self.class_names = list(datasets[0].CLASS_LABELS)

        if self.class_names:
            with torch.no_grad():
                self.class_text_embeds = self.text_encoder.encode(
                    self.class_names, device=self.device
                )

        # validation dataset info
        if datamodule is not None and datamodule.data_val is not None:
            for idx, dataset in enumerate(datamodule.data_val):
                self.val_datasets[idx] = dataset
                self.val_class_names[idx] = list(getattr(dataset, "CLASS_LABELS", []))
                self.val_subset_mapper[idx] = getattr(dataset, "subset_mapper", None)
                self.val_ignore_label[idx] = getattr(dataset, "ignore_label", -100)
                fg_idx = getattr(dataset, "fg_class_idx", None)
                self.val_fg_class_idx[idx] = (
                    [int(x) for x in fg_idx] if fg_idx is not None else []
                )

    def _get_val_postfix(self, dataloader_idx: int) -> str:
        dataset = self.val_datasets.get(dataloader_idx)
        if dataset is None:
            return f"val{dataloader_idx}"
        return str(
            getattr(dataset, "LOG_POSTFIX", None)
            or getattr(dataset, "dataset_name", None)
            or f"val{dataloader_idx}"
        )

    def _resolve_scene_names(
        self,
        batch: Dict[str, Any],
        dataloader_idx: int,
        num_samples: int,
    ) -> List[str]:
        scene_names: List[Optional[str]] = [None] * num_samples
        batch_scene_names = batch.get("scene_name")
        if isinstance(batch_scene_names, (list, tuple)):
            for sample_idx in range(min(len(batch_scene_names), num_samples)):
                raw_name = str(batch_scene_names[sample_idx]).strip()
                if raw_name:
                    scene_names[sample_idx] = raw_name

        missing_indices = [idx for idx, name in enumerate(scene_names) if name is None]
        if missing_indices:
            dataset = self.val_datasets.get(dataloader_idx)
            dataset_scene_names = list(getattr(dataset, "scene_names", [])) if dataset is not None else []
            scene_ptr = int(self._scene_ptr.get(dataloader_idx, 0))
            for offset_idx, sample_idx in enumerate(missing_indices):
                global_idx = scene_ptr + offset_idx
                if global_idx < len(dataset_scene_names):
                    scene_names[sample_idx] = str(dataset_scene_names[global_idx])
                else:
                    scene_names[sample_idx] = f"sample_{global_idx:06d}"
            self._scene_ptr[dataloader_idx] = scene_ptr + len(missing_indices)

        return [str(name) for name in scene_names]

    def _export_pred_labels(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        batch: Dict[str, Any],
        dataloader_idx: int,
    ) -> None:
        save_pred_root = os.environ.get("SAVE_PRED_LABEL_DIR", "").strip()
        if not save_pred_root:
            return

        offsets = batch.get("offset")
        if offsets is None:
            return
        if isinstance(offsets, torch.Tensor):
            offsets_cpu = offsets.detach().cpu().tolist()
        else:
            offsets_cpu = list(offsets)
        if len(offsets_cpu) < 2:
            return

        num_samples = len(offsets_cpu) - 1
        scene_names = self._resolve_scene_names(batch, dataloader_idx, num_samples)
        out_dir = Path(save_pred_root) / self._get_val_postfix(dataloader_idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        preds_np = preds.detach().cpu().numpy().astype(np.int32, copy=False)
        labels_np = labels.detach().cpu().numpy().astype(np.int32, copy=False)
        for sample_idx in range(num_samples):
            start = int(offsets_cpu[sample_idx])
            end = int(offsets_cpu[sample_idx + 1])
            if end <= start:
                continue
            scene_name = scene_names[sample_idx].replace("/", "__")
            np.save(out_dir / f"{scene_name}.pred.npy", preds_np[start:end])
            np.save(out_dir / f"{scene_name}.gt.npy", labels_np[start:end])

    def _extract_caption_texts(self, caption_data: Any) -> List[str]:
        texts: List[str] = []
        if caption_data is None:
            return texts
        for cap in caption_data:
            for cap_text in cap.get("caption", []):
                texts.append(str(cap_text))
        return texts

    def _extract_caption_embeds(self, caption_data: Any) -> List[torch.Tensor]:
        embeds: List[torch.Tensor] = []
        if caption_data is None:
            return embeds
        for cap in caption_data:
            for emb in cap.get("embed", []):
                if isinstance(emb, torch.Tensor):
                    embeds.append(emb)
                else:
                    embeds.append(torch.as_tensor(emb))
        return embeds

    def _build_text_guidance(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        if self.text_encoder is None:
            return None
        with torch.no_grad():
            cached_embeds = self._extract_caption_embeds(batch.get("caption_data"))
            if cached_embeds:
                embed_tensor = torch.stack(
                    [e.to(device=self.device, dtype=torch.float32) for e in cached_embeds], dim=0
                )
                return embed_tensor.mean(dim=0)
            captions = self._extract_caption_texts(batch.get("caption_data"))
            if captions:
                text_embeds = self.text_encoder.encode(captions, device=self.device)
                if text_embeds.ndim == 2 and text_embeds.shape[0] > 0:
                    return text_embeds.mean(dim=0)
            if self.class_text_embeds is not None and self.class_text_embeds.numel() > 0:
                return self.class_text_embeds.mean(dim=0)
        return None

    def forward(
        self,
        batch: Dict[str, Any],
        apply_tta_adapter: bool = False,
    ) -> Dict[str, torch.Tensor]:
        hpza_weight = self._hpza_loss_weight()
        use_text_guidance = hpza_weight > 0.0
        text_guidance = self._build_text_guidance(batch) if use_text_guidance else None
        if text_guidance is None:
            use_text_guidance = False
        need_text_feats = bool(use_text_guidance or self.zero_shot or (self.pmtl is not None))

        outputs = self.net(
            batch,
            text_guidance=text_guidance,
            apply_tta_adapter=apply_tta_adapter,
            return_text_feats=need_text_feats,
        )
        outputs["debug_use_text_guidance"] = torch.tensor(
            1.0 if use_text_guidance else 0.0,
            device=self.device,
        )
        return outputs

    def _compute_caption_alignment(self, text_feats, caption_data, offsets):
        if caption_data is None or self.text_encoder is None:
            return text_feats.sum() * 0.0

        cached_mask_feats = []
        cached_text_embeds = []
        online_mask_feats = []
        online_captions = []

        for b_idx, cap in enumerate(caption_data):
            base = int(offsets[b_idx])
            embed_list = cap.get("embed", [])
            for i, (idx_tensor, cap_text) in enumerate(zip(cap["idx"], cap["caption"])):
                if idx_tensor.numel() == 0:
                    continue
                global_idx = idx_tensor.to(text_feats.device) + base
                mask_feat = text_feats[global_idx].mean(dim=0)
                if i < len(embed_list):
                    emb = embed_list[i]
                    emb_t = emb if isinstance(emb, torch.Tensor) else torch.as_tensor(emb)
                    emb_t = emb_t.to(device=text_feats.device, dtype=mask_feat.dtype).view(-1)
                    if emb_t.numel() == mask_feat.numel():
                        cached_mask_feats.append(mask_feat)
                        cached_text_embeds.append(emb_t)
                    else:
                        online_mask_feats.append(mask_feat)
                        online_captions.append(str(cap_text))
                else:
                    online_mask_feats.append(mask_feat)
                    online_captions.append(str(cap_text))

        total_cnt = len(cached_mask_feats) + len(online_mask_feats)
        if total_cnt == 0:
            return text_feats.sum() * 0.0

        total_loss = text_feats.sum() * 0.0
        if cached_mask_feats:
            mask_feats = torch.stack(cached_mask_feats, dim=0)
            text_embeds = torch.stack(cached_text_embeds, dim=0)
            total_loss = total_loss + self.hpza_loss.caption_alignment(mask_feats, text_embeds) * (
                len(cached_mask_feats) / total_cnt
            )
        if online_mask_feats:
            mask_feats = torch.stack(online_mask_feats, dim=0)
            text_embeds = self.text_encoder.encode(online_captions, device=text_feats.device)
            total_loss = total_loss + self.hpza_loss.caption_alignment(mask_feats, text_embeds) * (
                len(online_mask_feats) / total_cnt
            )
        return total_loss

    def _compute_class_alignment(self, text_feats, labels):
        if labels is None or self.class_text_embeds is None:
            return text_feats.sum() * 0.0
        labels = labels.view(-1)
        text_feats = text_feats.view(labels.shape[0], -1)
        ignore_label = self.hparams.loss_cfg.get("ignore_label", -100)
        valid_mask = labels != ignore_label
        labels = labels[valid_mask]
        feats = text_feats[valid_mask]
        unique_ids = torch.unique(labels)
        if unique_ids.numel() == 0:
            return text_feats.sum() * 0.0
        class_feats = []
        class_text = []
        for cls_id in unique_ids:
            mask = labels == cls_id
            class_feats.append(feats[mask].mean(dim=0))
            if cls_id < self.class_text_embeds.shape[0]:
                class_text.append(self.class_text_embeds[int(cls_id)])
        if not class_feats or not class_text:
            return text_feats.sum() * 0.0
        class_feats = torch.stack(class_feats, dim=0)
        class_text = torch.stack(class_text, dim=0)
        return self.hpza_loss.class_alignment(class_feats, class_text)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        seg_logits = outputs["seg_logits"]
        inst_embeddings = outputs["inst_embeddings"]
        text_feats = outputs.get("text_feats")
        batch_size = self._batch_size_from_batch(batch)
        ignore_label = int(self.hparams.loss_cfg.get("ignore_label", -100))

        debug_use_text_guidance = float(outputs.get("debug_use_text_guidance", 0.0))
        self.log(
            "debug/use_text_guidance",
            debug_use_text_guidance,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        preds = torch.argmax(seg_logits, dim=1)
        seg_labels = batch.get("segment")
        train_stats = self._prediction_distribution_stats(preds, seg_labels, ignore_label)
        self.log(
            "debug/train_pred_unique_count",
            train_stats["pred_unique_count"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "debug/train_pred_top1_ratio",
            train_stats["pred_top1_ratio"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "debug/train_ignore_ratio",
            train_stats["ignore_ratio"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        if not self._input_stats_logged:
            input_stats = {
                "debug/input_coord_mean": outputs.get("debug_input_coord_mean"),
                "debug/input_coord_std": outputs.get("debug_input_coord_std"),
                "debug/input_color_mean": outputs.get("debug_input_color_mean"),
                "debug/input_color_std": outputs.get("debug_input_color_std"),
            }
            for metric_name, metric_val in input_stats.items():
                if metric_val is None:
                    continue
                if isinstance(metric_val, torch.Tensor):
                    metric_val = metric_val.detach().float()
                self.log(
                    metric_name,
                    metric_val,
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch_size,
                )
            self._input_stats_logged = True

        seg_sup_mask = batch.get("segment_supervision_mask")
        if seg_sup_mask is not None:
            self.log(
                "train/seg_sup_ratio",
                seg_sup_mask.float().mean(),
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        inst_sup_mask = batch.get("instance_supervision_mask")
        if inst_sup_mask is not None:
            self.log(
                "train/inst_sup_ratio",
                inst_sup_mask.float().mean(),
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

        if self._tdcl_active:
            seg_loss = self.seg_loss(
                seg_logits,
                batch.get("segment"),
                tail_feat=outputs.get("tail_feat"),
                prototypes=outputs.get("prototypes"),
                alpha=outputs.get("alpha"),
            )
            # Stream-accumulate a train confusion matrix so the TDCL head-
            # confuser table can be refreshed at the end of each epoch.
            seg_labels_conf = batch.get("segment")
            if seg_labels_conf is not None:
                num_classes = int(seg_logits.shape[1])
                if self._train_conf is None or self._train_conf.shape[0] != num_classes:
                    self._train_conf = torch.zeros(
                        (num_classes, num_classes), dtype=torch.long, device=self.device
                    )
                with torch.no_grad():
                    pred_conf = preds.view(-1).long()
                    lab_conf = seg_labels_conf.view(-1).long()
                    # Require both label and prediction to be inside the
                    # legal class range; clamping to 0 (the previous logic)
                    # would silently fold ignore-label / out-of-range entries
                    # into conf[0, 0] and pollute the head-confuser table.
                    valid_conf = (
                        (lab_conf != ignore_label)
                        & (lab_conf >= 0)
                        & (lab_conf < num_classes)
                        & (pred_conf >= 0)
                        & (pred_conf < num_classes)
                    )
                    if bool(valid_conf.any()):
                        flat = lab_conf[valid_conf] * num_classes + pred_conf[valid_conf]
                        binc = torch.bincount(flat, minlength=num_classes * num_classes)
                        self._train_conf += binc.view(num_classes, num_classes)
        else:
            seg_loss = self.seg_loss(seg_logits, batch.get("segment"))
        inst_loss = self.inst_loss(inst_embeddings, batch.get("instance"))

        weights = self.hparams.loss_cfg.get("weights", {})
        if self.pmtl is not None:
            weights = self.pmtl.get_weights(self.current_epoch)
        hpza_weight = float(weights.get("hpza_loss", 1.0))

        if hpza_weight > 0.0:
            if text_feats is None:
                point_feats = outputs.get("point_feats")
                if point_feats is not None and hasattr(self.net, "text_proj"):
                    text_feats = self.net.text_proj(point_feats)
                else:
                    raise RuntimeError(
                        "HPZA loss is enabled but text features are unavailable."
                    )
            hpza_caption = self._compute_caption_alignment(
                text_feats, batch.get("caption_data"), batch["offset"]
            )
            hpza_class = self._compute_class_alignment(text_feats, batch.get("segment"))
            hpza_loss = hpza_caption + hpza_class
        else:
            hpza_loss = seg_loss.new_zeros(())

        loss = (
            seg_loss * float(weights.get("seg_loss", 1.0))
            + inst_loss * float(weights.get("instance_loss", 1.0))
            + hpza_loss * hpza_weight
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/seg_loss", seg_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/inst_loss", inst_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/hpza_loss", hpza_loss, on_step=True, on_epoch=True, batch_size=batch_size)

        # TDP prototype EMA update happens after the loss has been computed so
        # the back-prop graph is built off the *current* prototypes, then the
        # next iteration sees a fresher snapshot.
        if self._tdcl_active and hasattr(self.net, "decoder") and hasattr(
            self.net.decoder, "ema_update"
        ):
            tail_feat = outputs.get("tail_feat")
            seg_labels = batch.get("segment")
            if tail_feat is not None and seg_labels is not None:
                self.net.decoder.ema_update(
                    tail_feat=tail_feat.detach(),
                    labels=seg_labels.detach(),
                    ignore_label=ignore_label,
                )

        return loss

    def on_train_epoch_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(int(self.current_epoch))
        # Reset the streaming train confusion buffer at epoch start.
        if self._tdcl_active:
            self._train_conf = None

    def on_train_epoch_end(self) -> None:
        # Refresh TDCL head-confuser table from the just-finished epoch.
        if (
            self._tdcl_active
            and hasattr(self.seg_loss, "update_head_confuser")
            and self._train_conf is not None
        ):
            try:
                updated = self.seg_loss.update_head_confuser(self._train_conf)
                self.log(
                    "debug/tdcl_head_confuser_updated",
                    float(updated),
                    on_step=False,
                    on_epoch=True,
                )
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(f"TDCL head-confuser update failed: {exc}")

    def on_validation_epoch_start(self) -> None:
        self._restore_eval_bn_batch_stats_mode()
        self.val_conf = {}
        for idx, names in self.val_class_names.items():
            num_classes = len(names)
            self.val_conf[idx] = torch.zeros(
                (num_classes, num_classes), dtype=torch.int64, device=self.device
            )
            self._scene_ptr[idx] = 0
        self._set_eval_bn_batch_stats_mode()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        if self.zero_shot and self.class_text_embeds is not None:
            outputs = self(batch)
            text_feats = F.normalize(outputs["text_feats"], dim=-1)
            class_text = F.normalize(self.class_text_embeds, dim=-1)
            seg_logits = text_feats @ class_text.t()
            debug_use_text_guidance = float(outputs.get("debug_use_text_guidance", 0.0))
        elif self.tta_evaluator is not None:
            seg_logits = self.tta_evaluator(self, batch)
            debug_use_text_guidance = float(self._hpza_loss_weight() > 0.0)
        else:
            outputs = self(batch)
            seg_logits = outputs["seg_logits"]
            debug_use_text_guidance = float(outputs.get("debug_use_text_guidance", 0.0))

        if "segment" not in batch:
            return
        preds = torch.argmax(seg_logits, dim=1)
        labels = batch["segment"]
        batch_size = self._batch_size_from_batch(batch)
        ignore_label = int(self.val_ignore_label.get(dataloader_idx, -100))

        self.log(
            "debug/use_text_guidance",
            debug_use_text_guidance,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"debug/use_text_guidance_{dataloader_idx}",
            debug_use_text_guidance,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        val_stats = self._prediction_distribution_stats(preds, labels, ignore_label)
        self.log(
            f"debug/val_pred_unique_count_{dataloader_idx}",
            val_stats["pred_unique_count"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"debug/val_pred_top1_ratio_{dataloader_idx}",
            val_stats["pred_top1_ratio"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"debug/val_ignore_ratio_{dataloader_idx}",
            val_stats["ignore_ratio"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self._export_pred_labels(preds, labels, batch, dataloader_idx)
        num_classes = len(self.val_class_names.get(dataloader_idx, []))
        if num_classes == 0:
            return
        conf = self.val_conf[dataloader_idx]
        self.val_conf[dataloader_idx] = update_confusion_matrix(
            conf,
            preds,
            labels,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )

    def on_validation_epoch_end(self) -> None:
        epoch_per_class = {}
        epoch_dead_classes = {}
        epoch_metrics = {}
        primary_postfix = None

        for idx, conf in self.val_conf.items():
            num_classes = conf.shape[0]
            if num_classes == 0:
                continue
            conf_float = conf.float()
            iou = compute_iou_from_conf(conf_float)
            miou = float(iou.mean().item())

            row_sum = conf_float.sum(dim=1)
            col_sum = conf_float.sum(dim=0)
            union = row_sum + col_sum - torch.diag(conf_float)

            present_gt_mask = row_sum > 0
            present_union_mask = union > 0

            if present_gt_mask.any():
                miou_present_gt = float(iou[present_gt_mask].mean().item())
            else:
                miou_present_gt = 0.0

            if present_union_mask.any():
                miou_present_union = float(iou[present_union_mask].mean().item())
            else:
                miou_present_union = 0.0

            fg_idx = [
                int(class_idx)
                for class_idx in self.val_fg_class_idx.get(idx, [])
                if 0 <= int(class_idx) < num_classes
            ]
            if len(fg_idx) == 0:
                class_names = self.val_class_names.get(idx, [])
                fg_idx = [
                    class_idx
                    for class_idx, class_name in enumerate(class_names)
                    if str(class_name) not in {"wall", "floor", "ceiling"}
                    and "other" not in str(class_name)
                ]
            if len(fg_idx) == 0:
                fg_idx = list(range(num_classes))
            miou_fg_mosaic = float(iou[fg_idx].mean().item())

            subset_vals = compute_subset_miou(
                iou,
                self.val_class_names[idx],
                self.val_subset_mapper.get(idx),
            )
            postfix = f"{idx}"
            class_names = self.val_class_names.get(idx, [])
            row_sum_cpu = row_sum.detach().cpu()
            col_sum_cpu = col_sum.detach().cpu()
            iou_cpu = iou.detach().cpu()
            class_entries = []
            dead_entries = []
            for class_idx in range(num_classes):
                class_name = (
                    str(class_names[class_idx])
                    if class_idx < len(class_names)
                    else f"class_{class_idx}"
                )
                gt_count = int(row_sum_cpu[class_idx].item())
                pred_count = int(col_sum_cpu[class_idx].item())
                class_iou = float(iou_cpu[class_idx].item())
                item = {
                    "class_id": int(class_idx),
                    "class_name": class_name,
                    "iou": class_iou,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                }
                class_entries.append(item)
                if (gt_count > 0) and (class_iou <= 1e-12):
                    dead_entries.append(item)
            epoch_per_class[postfix] = class_entries
            epoch_dead_classes[postfix] = dead_entries
            epoch_metrics[postfix] = {
                "miou": miou,
                "miou_present_gt": miou_present_gt,
                "miou_present_union": miou_present_union,
                "miou_fg_mosaic": miou_fg_mosaic,
            }

            self.log(f"val/miou_{postfix}", miou, prog_bar=True)
            self.log(f"val/miou_present_gt_{postfix}", miou_present_gt)
            self.log(f"val/miou_present_union_{postfix}", miou_present_union)
            self.log(f"val/miou_fg_mosaic_{postfix}", miou_fg_mosaic)
            if idx == 0:
                primary_postfix = postfix
                self.log("val/miou", miou, prog_bar=True)
                self.log("val/miou_present_gt", miou_present_gt)
                self.log("val/miou_present_union", miou_present_union)
                self.log("val/miou_fg_mosaic", miou_fg_mosaic)
            for k, v in subset_vals.items():
                self.log(f"val/{k}_{postfix}", v)
                if k.endswith("_miou"):
                    self.log(f"val/{k[:-5]}_{postfix}", v)

        if not epoch_per_class:
            self._restore_eval_bn_batch_stats_mode()
            return

        if primary_postfix is None:
            primary_postfix = sorted(epoch_per_class.keys())[0]
        primary_metrics = epoch_metrics.get(primary_postfix, {})

        metric_key = str(self.best_metric)
        if metric_key.startswith("val/"):
            metric_key = metric_key[len("val/") :]
        current_metric = float(primary_metrics.get(metric_key, primary_metrics.get("miou", 0.0)))

        diag_payload = {
            "epoch": int(self.current_epoch),
            "best_metric_name": str(self.best_metric),
            "best_metric_value": current_metric,
            "primary_postfix": str(primary_postfix),
            "metrics_by_dataloader": epoch_metrics,
            "per_class_by_dataloader": epoch_per_class,
            "dead_classes_by_dataloader": epoch_dead_classes,
            "primary_per_class": epoch_per_class.get(primary_postfix, []),
            "primary_dead_classes": epoch_dead_classes.get(primary_postfix, []),
        }
        self._last_val_diag = diag_payload

        if current_metric >= self._best_val_metric:
            self._best_val_metric = current_metric
            self._best_val_diag = diag_payload
            output_dir = self._resolve_output_dir()
            self._write_json(
                output_dir / "per_class_iou_best.json",
                {
                    "epoch": diag_payload["epoch"],
                    "best_metric_name": diag_payload["best_metric_name"],
                    "best_metric_value": diag_payload["best_metric_value"],
                    "primary_postfix": diag_payload["primary_postfix"],
                    "by_dataloader": diag_payload["per_class_by_dataloader"],
                    "primary": diag_payload["primary_per_class"],
                },
            )
            self._write_json(
                output_dir / "dead_classes_best.json",
                {
                    "epoch": diag_payload["epoch"],
                    "best_metric_name": diag_payload["best_metric_name"],
                    "best_metric_value": diag_payload["best_metric_value"],
                    "primary_postfix": diag_payload["primary_postfix"],
                    "by_dataloader": diag_payload["dead_classes_by_dataloader"],
                    "primary": diag_payload["primary_dead_classes"],
                },
            )
        self._restore_eval_bn_batch_stats_mode()

    def on_fit_end(self) -> None:
        if self._last_val_diag is None:
            return
        output_dir = self._resolve_output_dir()
        diag_payload = self._last_val_diag
        self._write_json(
            output_dir / "per_class_iou_last.json",
            {
                "epoch": diag_payload["epoch"],
                "best_metric_name": diag_payload["best_metric_name"],
                "best_metric_value": diag_payload["best_metric_value"],
                "primary_postfix": diag_payload["primary_postfix"],
                "by_dataloader": diag_payload["per_class_by_dataloader"],
                "primary": diag_payload["primary_per_class"],
            },
        )
        self._write_json(
            output_dir / "dead_classes_last.json",
            {
                "epoch": diag_payload["epoch"],
                "best_metric_name": diag_payload["best_metric_name"],
                "best_metric_value": diag_payload["best_metric_value"],
                "primary_postfix": diag_payload["primary_postfix"],
                "by_dataloader": diag_payload["dead_classes_by_dataloader"],
                "primary": diag_payload["primary_dead_classes"],
            },
        )

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        if self.hparams.scheduler is None:
            return optimizer

        scheduler_cfg = self.hparams.scheduler
        if isinstance(scheduler_cfg, DictConfig):
            scheduler_cfg = OmegaConf.create(
                OmegaConf.to_container(scheduler_cfg, resolve=True)
            )

        target = str(scheduler_cfg.get("_target_", ""))
        if target.endswith("OneCycleLR"):
            auto_total_steps = bool(scheduler_cfg.get("auto_total_steps", True))
            if auto_total_steps:
                scheduler_cfg["total_steps"] = max(int(self.trainer.estimated_stepping_batches), 1)
            if "auto_total_steps" in scheduler_cfg:
                del scheduler_cfg["auto_total_steps"]

        scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.scheduler_interval,
            },
        }
