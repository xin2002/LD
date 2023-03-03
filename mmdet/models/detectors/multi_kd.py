import mmcv
from numpy.lib.twodim_base import tri
import torch
from mmcv.runner import load_checkpoint

from .. import build_detector
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class MultiTeacherKnowledgeDistillationSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_configs (list[str | dict]): Config file path
            or the config object of teacher models.
        teacher_ckpt (str, optional): Checkpoint path of teacher models.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_configs,
                 output_feature=False,
                 teacher_ckpts=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert len(teacher_configs) >= 2

        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        self.output_feature = output_feature
        # Build teacher model
        self.teacher_models = []
        for teacher_config in teacher_configs:
            if isinstance(teacher_config, str):
                teacher_config = mmcv.Config.fromfile(teacher_config)
            self.teacher_models.append(build_detector(teacher_config['model']))
        if teacher_ckpts is not None:
            for ckpt_index in range(len(teacher_ckpts)):
                load_checkpoint(
                    self.teacher_models[ckpt_index], teacher_ckpts[ckpt_index], map_location='cpu')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        
        with torch.no_grad():
            teacher_x = []
            out_teacher = []
            for teacher_ind in range(len(self.teacher_models)):
                single_teacher_x = self.teacher_models[teacher_ind].extract_feat(img)
                single_out_teacher = self.teacher_models[teacher_ind].bbox_head(single_teacher_x)
                teacher_x.append(single_teacher_x)
                out_teacher.append(single_out_teacher)
        if not self.output_feature:
            losses = self.bbox_head.forward_train(x, out_teacher, img_metas,
                                                  gt_bboxes, gt_labels,
                                                  gt_bboxes_ignore)
        else:
            losses = self.bbox_head.forward_train(x, out_teacher, teacher_x,
                                                  img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
        return losses

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        for teacher_model in self.teacher_models:
                teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            for teacher_model in self.teacher_models:
                teacher_model.train(False)
        else:
            for teacher_model in self.teacher_models:
                teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_models':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
