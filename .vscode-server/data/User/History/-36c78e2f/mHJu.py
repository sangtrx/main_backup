# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_softmax, TransformerEncoder
from .bmn import BoundaryMatchingNetwork
# from .gtad import GTAD


class EventDetection(nn.Module):
    def __init__(self, cfg):
        super(EventDetection, self).__init__()
        self.use_env_linear = cfg.MODEL.ENV_HIDDEN_DIM is not None
        self.use_agent_linear = cfg.MODEL.AGENT_HIDDEN_DIM is not None

        if self.use_env_linear:
            self.env_linear = nn.Linear(cfg.MODEL.ENV_DIM, cfg.MODEL.ENV_HIDDEN_DIM)
        if self.use_agent_linear:
            self.agent_linear = nn.Linear(cfg.MODEL.AGENT_DIM, cfg.MODEL.AGENT_HIDDEN_DIM)

        self.agents_fuser = TransformerEncoder(cfg)
        self.agents_environment_fuser = TransformerEncoder(cfg)

        self.bmm_name = cfg.MODEL.BOUNDARY_MATCHING_MODULE
        if self.bmm_name == 'bmn':
            self.event_detector = BoundaryMatchingNetwork(cfg)
        elif self.bmm_name == 'gtad':
            self.event_detector = GTAD(cfg)

        self.attention_steps = cfg.TRAIN.ATTENTION_STEPS
        self.topk_hard_attention = cfg.MODEL.TOPK_AGENTS

    def fuse_agent(self, agent_feats, agent_masks, env_feats):
        bsz, tmprl_sz, n_boxes, ft_sz = agent_feats.size()
        step = self.attention_steps

        agent_env_feats = torch.unsqueeze(env_feats, 2) + agent_feats
        # Fuse all agents together at every temporal point
        smpl_bgn = 0

        agent_fused_features = torch.zeros(bsz, tmprl_sz, ft_sz).cuda()
        if n_boxes == 0:
            return agent_fused_features

        for smpl_bgn in range(0, tmprl_sz, step):
            smpl_end = smpl_bgn + step

            ae_feats = agent_env_feats[:, smpl_bgn:smpl_end].contiguous().view(-1, n_boxes, ft_sz)  # bsz x n_boxes x feat_dim
            masks = agent_masks[:, smpl_bgn:smpl_end].contiguous().view(-1, n_boxes)  # bsz x n_boxes

            #hard_attn_masks = masks
            l2_norm = torch.norm(ae_feats, dim=-1)  # bsz x n_boxes
            l2_norm_softmax = masked_softmax(l2_norm, masks)  # bsz x n_boxes

            # Adaptive threshold is 1 / number of bounding boxes:
            ada_thresh = torch.clamp(1. / torch.sum(masks, dim=-1, keepdim=True), 0., 1.)

            # Generate hard attention masks
            hard_attn_masks = l2_norm_softmax >= ada_thresh  # bsz x n_boxes

            '''
            # Generate masks for topk l2_norm
            _, topk_indices = torch.topk(l2_norm_softmax, self.topk_hard_attention, dim=-1)
            topk_masks = torch.zeros_like(hard_attn_masks, dtype=torch.bool)
            topk_masks = topk_masks.scatter(-1, topk_indices, 1)

            # merge hard attention masks with topk masks to only keep topk elements higher than adaptive threshold
            hard_attn_masks = torch.logical_and(hard_attn_masks, topk_masks)

            hard_attn_count = torch.sum(hard_attn_masks, dim=-1).cpu().numpy()  # bsz
            mask_count = torch.sum(masks, dim=-1).cpu().numpy()  # bsz
            '''

            keep_mask = (torch.sum(hard_attn_masks, dim=-1) > 0)  # bsz
            keep_indices = torch.masked_select(torch.arange(hard_attn_masks.size(0)).cuda(), keep_mask)  # keep_mask

            fuser_input = agent_feats[:, smpl_bgn:smpl_end].contiguous().view(-1, n_boxes, ft_sz).permute(1, 0, 2)  # n_boxes x bsz x feat_dim

            if len(keep_indices) > 0:
                fuser_input = fuser_input[:, keep_indices]  # n_boxes x keep_mask x feat_dim
                hard_attn_masks = hard_attn_masks[keep_indices]  # keep_mask x n_boxes

                padded_output = torch.zeros(bsz * (smpl_end - smpl_bgn), ft_sz).cuda()  # bsz x feat_dim
                fuser_output = self.agents_fuser(fuser_input, key_padding_mask=~hard_attn_masks)  # n_boxes x keep_mask x feat_dim
                #fuser_output = fuser_input * hard_attn_masks.permute(1, 0).contiguous().unsqueeze(-1)
                fuser_output = torch.sum(fuser_output, dim=0) / torch.sum(hard_attn_masks, dim=-1, keepdim=True)  # keep_mask x feat_dim
                padded_output[keep_indices] = fuser_output
                agent_fused_features[:, smpl_bgn:smpl_end] = padded_output.view(bsz, -1, ft_sz)

        return agent_fused_features

    def forward(self, env_features=None, agent_features=None, agent_masks=None):
        if self.use_env_linear and env_features is not None:
            env_features = self.env_linear(env_features)
        if self.use_agent_linear and agent_features is not None:
            agent_features = self.agent_linear(agent_features)

        if agent_features is None:
            return self.event_detector(env_features.permute(0, 2, 1))

        agent_fused_features = self.fuse_agent(agent_features, agent_masks, env_features)

        if env_features is None:
            return self.event_detector(agent_fused_features.permute(0, 2, 1))

        env_agent_cat_features = torch.stack([env_features, agent_fused_features], dim=2)

        bsz, tmprl_sz, ft_sz = env_features.shape
        step = self.attention_steps
        smpl_bgn = 0
        context_features = torch.zeros(bsz, tmprl_sz, ft_sz).cuda()

        for smpl_bgn in range(0, tmprl_sz, step):
            smpl_end = smpl_bgn + step

            fuser_input = env_agent_cat_features[:, smpl_bgn:smpl_end].contiguous()
            fuser_input = fuser_input.view(-1, 2, ft_sz).permute(1, 0, 2)

            fuser_output = self.agents_environment_fuser(fuser_input)
            fuser_output = torch.mean(fuser_output, dim=0)
            #fuser_output = fuser_input
            #fuser_output = torch.mean(fuser_output, dim=0)
            context_features[:, smpl_bgn:smpl_end] = fuser_output.view(bsz, -1, ft_sz)

        return self.event_detector(context_features.permute(0, 2, 1))
