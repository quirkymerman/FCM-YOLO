"""Training-only knowledge distillation loss for MGD-YOLO.

The teacher is used only during training. This file does not change the student
network architecture, exported model, or inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.tal import make_anchors


@dataclass
class KDConfig:
    alpha_kd: float = 0.5
    lambda_dfl: float = 0.5
    lambda_score: float = 1.0
    small_thr: float = 1024.0
    warmup_epochs: int = 3
    student_imgsz: int = 640
    teacher_imgsz: int = 960
    conf_thres: float = 0.25
    iou_thres: float = 0.7
    max_det: int = 300
    kd_mode: str = "full"  # full | score | loc | plain
    topk_anchors: int = 9


class SmallObjectAwareKD:
    """Wrap the original detection criterion and add sparse teacher guidance.

    L_total = L_det + alpha(t) * (lambda_score * L_score + lambda_dfl * L_box)

    First version uses bbox-level localization distillation as a stable fallback.
    TODO: add DFL KL only when teacher/student grids are explicitly aligned.
    """

    def __init__(
        self,
        base_criterion,
        teacher_model: torch.nn.Module,
        cfg: KDConfig,
        get_epoch: Callable[[], int] | None = None,
    ):
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.cfg = cfg
        self.get_epoch = get_epoch or (lambda: 0)
        self.device = base_criterion.device
        self.nc = base_criterion.nc
        self.no = base_criterion.no
        self.reg_max = base_criterion.reg_max
        self.stride = base_criterion.stride
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, preds, batch):
        det_loss, det_items = self.base_criterion(preds, batch)

        # Validation runs under no_grad. Keep a six-column loss tensor but do
        # not execute the teacher or add KD loss during validation.
        if not torch.is_grad_enabled():
            zeros = torch.zeros(3, device=self.device, dtype=det_items.dtype)
            return det_loss, torch.cat((det_items.detach(), zeros))

        kd_score, kd_box = self.compute_kd(preds, batch)
        # Hard guard: KD must never poison the original detection loss.
        kd_score = torch.nan_to_num(kd_score, nan=0.0, posinf=0.0, neginf=0.0)
        kd_box = torch.nan_to_num(kd_box, nan=0.0, posinf=0.0, neginf=0.0)
        kd_raw = self.cfg.lambda_score * kd_score + self.cfg.lambda_dfl * kd_box
        kd_raw = torch.nan_to_num(kd_raw, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = self.alpha()
        total = det_loss + alpha * kd_raw
        kd_items = torch.stack((alpha * kd_score.detach(), alpha * kd_box.detach(), alpha * kd_raw.detach()))
        return total, torch.cat((det_items.detach(), kd_items))

    def alpha(self) -> float:
        warmup = max(int(self.cfg.warmup_epochs), 1)
        current_epoch = int(self.get_epoch()) + 1
        return float(self.cfg.alpha_kd) * min(1.0, current_epoch / warmup)

    def compute_kd(self, preds, batch):
        feats = self._raw_feats(preds)
        if feats is None:
            z = torch.zeros((), device=self.device)
            return z, z

        pred_scores, pred_boxes, anchor_xy = self._student_outputs(feats)
        teacher_dets = self._teacher_detections(batch["img"])

        if self.cfg.kd_mode == "plain":
            return self._sparse_kd_from_teacher(pred_scores, pred_boxes, anchor_xy, teacher_dets, False)

        use_score = self.cfg.kd_mode in {"full", "score"}
        use_box = self.cfg.kd_mode in {"full", "loc"}
        kd_score, kd_box = self._sparse_kd_from_teacher(pred_scores, pred_boxes, anchor_xy, teacher_dets, True)
        if not use_score:
            kd_score = kd_score.detach() * 0.0
        if not use_box:
            kd_box = kd_box.detach() * 0.0
        return kd_score, kd_box

    @staticmethod
    def _raw_feats(preds):
        if isinstance(preds, tuple):
            preds = preds[1]
        if isinstance(preds, dict):
            preds = preds.get("one2many", preds.get("one2one"))
        if isinstance(preds, (list, tuple)) and preds and all(torch.is_tensor(x) for x in preds):
            return preds
        return None

    def _student_outputs(self, feats: Iterable[torch.Tensor]):
        feats = list(feats)
        pred_distri, pred_scores = torch.cat([x.view(feats[0].shape[0], self.no, -1) for x in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        pred_boxes = self.base_criterion.bbox_decode(anchor_points, pred_distri) * stride_tensor
        anchor_xy = anchor_points * stride_tensor
        return pred_scores, pred_boxes, anchor_xy

    @torch.no_grad()
    def _teacher_detections(self, imgs: torch.Tensor):
        teacher_imgs = imgs
        if int(self.cfg.teacher_imgsz) > 0 and imgs.shape[-1] != int(self.cfg.teacher_imgsz):
            teacher_imgs = F.interpolate(
                imgs,
                size=(int(self.cfg.teacher_imgsz), int(self.cfg.teacher_imgsz)),
                mode="bilinear",
                align_corners=False,
            )
        out = self.teacher_model(teacher_imgs)
        pred = out[0] if isinstance(out, tuple) else out
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        dets = non_max_suppression(
            pred,
            conf_thres=float(self.cfg.conf_thres),
            iou_thres=float(self.cfg.iou_thres),
            max_det=int(self.cfg.max_det),
            nc=self.nc,
        )
        return [d.detach() for d in dets]

    def _sparse_kd_from_teacher(self, pred_scores, pred_boxes, anchor_xy, teacher_dets, use_small_weight: bool):
        device = pred_scores.device
        z = torch.zeros((), device=device, dtype=pred_scores.dtype)
        score_terms, box_terms, weight_terms = [], [], []
        bs, _, nc = pred_scores.shape

        for b in range(min(bs, len(teacher_dets))):
            det = teacher_dets[b]
            if det.numel() == 0:
                continue

            sx = pred_scores.new_tensor(float(self.cfg.student_imgsz))
            sy = pred_scores.new_tensor(float(self.cfg.student_imgsz))
            tx = pred_scores.new_tensor(float(self.cfg.teacher_imgsz))
            ty = pred_scores.new_tensor(float(self.cfg.teacher_imgsz))
            boxes = det[:, :4].to(device=device, dtype=pred_scores.dtype).clone()
            boxes[:, [0, 2]] *= sx / tx
            boxes[:, [1, 3]] *= sy / ty
            boxes = torch.nan_to_num(boxes, nan=0.0, posinf=float(self.cfg.student_imgsz), neginf=0.0)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, float(self.cfg.student_imgsz))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, float(self.cfg.student_imgsz))
            valid_boxes = (boxes[:, 2] > boxes[:, 0] + 1.0) & (boxes[:, 3] > boxes[:, 1] + 1.0)
            if not valid_boxes.any():
                continue
            boxes = boxes[valid_boxes]
            conf = det[:, 4].to(device=device, dtype=pred_scores.dtype).clamp_(0, 1)
            conf = conf[valid_boxes]
            cls = det[:, 5].to(device=device, dtype=torch.long).clamp_(0, nc - 1)[valid_boxes]

            areas = ((boxes[:, 2] - boxes[:, 0]).clamp_min(1.0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0))
            mean_area = areas.mean().clamp_min(1.0)
            scale_w = torch.sqrt(mean_area / areas).clamp(1.0, 3.0) if use_small_weight else torch.ones_like(areas)
            if use_small_weight:
                scale_w = torch.where(areas <= float(self.cfg.small_thr), scale_w, torch.ones_like(scale_w))

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            for j in range(boxes.shape[0]):
                box = boxes[j]
                inside = (
                    (anchor_xy[:, 0] >= box[0])
                    & (anchor_xy[:, 0] <= box[2])
                    & (anchor_xy[:, 1] >= box[1])
                    & (anchor_xy[:, 1] <= box[3])
                )
                idx = inside.nonzero(as_tuple=False).flatten()
                if idx.numel() == 0:
                    dist = (anchor_xy - centers[j]).pow(2).sum(1)
                    idx = dist.topk(k=1, largest=False).indices
                elif idx.numel() > int(self.cfg.topk_anchors):
                    dist = (anchor_xy[idx] - centers[j]).pow(2).sum(1)
                    idx = idx[dist.topk(k=int(self.cfg.topk_anchors), largest=False).indices]

                c = cls[j]
                target = conf[j].expand_as(idx).to(pred_scores.dtype)
                w = (conf[j] * scale_w[j]).expand_as(target).to(pred_scores.dtype)
                score_terms.append(self.bce(pred_scores[b, idx, c], target) * w)

                pred_box = torch.nan_to_num(
                    pred_boxes[b, idx].float(), nan=0.0, posinf=float(self.cfg.student_imgsz), neginf=0.0
                )
                pred_box[:, [0, 2]] = pred_box[:, [0, 2]].clamp(0, float(self.cfg.student_imgsz))
                pred_box[:, [1, 3]] = pred_box[:, [1, 3]].clamp(0, float(self.cfg.student_imgsz))
                ref_box = box.float().view(1, 4)
                iou = bbox_iou(ref_box, pred_box, xywh=False, CIoU=True).view(-1)
                iou_loss = torch.nan_to_num(1.0 - iou, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 2.0)
                l1_loss = F.l1_loss(pred_box, ref_box.expand(idx.numel(), 4), reduction="none").mean(1)
                l1_loss = torch.nan_to_num(l1_loss, nan=0.0, posinf=float(self.cfg.student_imgsz), neginf=0.0)
                box_terms.append(torch.nan_to_num((iou_loss + 0.02 * l1_loss) * w, nan=0.0, posinf=0.0, neginf=0.0))
                weight_terms.append(w)

        if not score_terms:
            return z, z

        denom = torch.cat(weight_terms).sum().clamp_min(1.0)
        return torch.cat(score_terms).sum() / denom, torch.cat(box_terms).sum() / denom
