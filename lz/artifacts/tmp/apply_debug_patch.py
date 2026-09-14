from pathlib import Path

p = Path('/root/lz/src/models/lightning_modules/lz_module.py')
text = p.read_text(encoding='utf-8')

old1 = '''        inst_sup_mask = batch.get("instance_supervision_mask")
        if inst_sup_mask is not None:
            self.log(
                "train/inst_sup_ratio",
                inst_sup_mask.float().mean(),
                on_step=True,
                on_epoch=True,
            )

        seg_loss = self.seg_loss(seg_logits, batch.get("segment"))
'''

new1 = '''        inst_sup_mask = batch.get("instance_supervision_mask")
        if inst_sup_mask is not None:
            self.log(
                "train/inst_sup_ratio",
                inst_sup_mask.float().mean(),
                on_step=True,
                on_epoch=True,
            )

        if batch_idx % 20 == 0 and "segment" in batch:
            labels_dbg = batch["segment"]
            ignore_label = getattr(self.seg_loss, "ignore_label", -100)
            valid_dbg = labels_dbg != ignore_label
            gt_min = float(labels_dbg[valid_dbg].min().item()) if valid_dbg.any() else -1.0
            gt_max = float(labels_dbg[valid_dbg].max().item()) if valid_dbg.any() else -1.0
            ignore_ratio = float((~valid_dbg).float().mean().item())
            preds_dbg = torch.argmax(seg_logits.detach(), dim=1)
            pred_vals, pred_counts = torch.unique(preds_dbg, return_counts=True)
            pred_unique_count = float(pred_vals.numel())
            pred_top1_ratio = float(pred_counts.max().float().item() / max(int(preds_dbg.numel()), 1))
            self.log("debug/train_gt_min", gt_min, on_step=True, on_epoch=False)
            self.log("debug/train_gt_max", gt_max, on_step=True, on_epoch=False)
            self.log("debug/train_ignore_ratio", ignore_ratio, on_step=True, on_epoch=False)
            self.log("debug/train_pred_unique_count", pred_unique_count, on_step=True, on_epoch=False)
            self.log("debug/train_pred_top1_ratio", pred_top1_ratio, on_step=True, on_epoch=False)

        seg_loss = self.seg_loss(seg_logits, batch.get("segment"))
'''

old2 = '''        preds = torch.argmax(seg_logits, dim=1)
        labels = batch["segment"]
        self._export_pred_labels(preds, labels, batch, dataloader_idx)
'''

new2 = '''        preds = torch.argmax(seg_logits, dim=1)
        labels = batch["segment"]
        if batch_idx % 20 == 0:
            ignore_label = self.val_ignore_label.get(dataloader_idx, -100)
            valid_dbg = labels != ignore_label
            gt_min = float(labels[valid_dbg].min().item()) if valid_dbg.any() else -1.0
            gt_max = float(labels[valid_dbg].max().item()) if valid_dbg.any() else -1.0
            ignore_ratio = float((~valid_dbg).float().mean().item())
            pred_vals, pred_counts = torch.unique(preds, return_counts=True)
            pred_unique_count = float(pred_vals.numel())
            pred_top1_ratio = float(pred_counts.max().float().item() / max(int(preds.numel()), 1))
            self.log(f"debug/val_gt_min_{dataloader_idx}", gt_min, on_step=True, on_epoch=False)
            self.log(f"debug/val_gt_max_{dataloader_idx}", gt_max, on_step=True, on_epoch=False)
            self.log(f"debug/val_ignore_ratio_{dataloader_idx}", ignore_ratio, on_step=True, on_epoch=False)
            self.log(f"debug/val_pred_unique_count_{dataloader_idx}", pred_unique_count, on_step=True, on_epoch=False)
            self.log(f"debug/val_pred_top1_ratio_{dataloader_idx}", pred_top1_ratio, on_step=True, on_epoch=False)
        self._export_pred_labels(preds, labels, batch, dataloader_idx)
'''

if old1 not in text:
    raise SystemExit('patch block 1 not found')
if old2 not in text:
    raise SystemExit('patch block 2 not found')

text = text.replace(old1, new1, 1)
text = text.replace(old2, new2, 1)
p.write_text(text, encoding='utf-8')
print('patched', p)
