# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepScale Team

import math
import torch
from torch.autograd import Function
import torch.nn as nn
from packaging import version as pkg_version
from deepscale.utils.logging import log_dist
from deepscale.accelerator import get_accelerator
from deepscale.ops.transformer.inference.op_binding.workspace import WorkspaceOp
from deepscale.ops.transformer.inference.op_binding.softmax_context import SoftmaxContextOp
from deepscale.ops.transformer.inference.op_binding import LinearOp
from deepscale.ops.transformer.inference.op_binding.pad_transform import PadTransformOp

minus_inf = -10000.0
triton_flash_attn = None


def load_triton_flash_attn():
    global triton_flash_attn
    try:
        import triton
    except ImportError:
        raise ImportError("Please install triton 2.0+ or `pip install deepscale[sd]`")

    if pkg_version.parse(triton.__version__) < pkg_version.parse("2.0"):
        raise ImportError("Please install triton 2.0+ or `pip install deepscale[sd]`")

    from .triton_ops import triton_flash_attn


class DeepScaleDiffusersAttentionFunction(Function):

    @staticmethod
    def forward(ctx, input, context, input_mask, config, attn_qkvw, attn_qw, attn_kw, attn_vw, attn_qkvb,
                num_attention_heads_per_partition, norm_factor, hidden_size_per_partition, attn_ow, attn_ob,
                do_out_bias, score_context_func, linear_func, pad_transform_func, triton_flash_attn_kernel,
                rope_theta):

        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3)
            new_x_layer_shape = x.size()[:-2] + \
                                      (hidden_size_per_partition,)
            return x.reshape(*new_x_layer_shape)

        def _transpose_for_scores(x):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (num_attention_heads_per_partition, attention_head_size)
            x = x.reshape(*new_x_shape)
            x = x.permute(0, 2, 1, 3)
            return x.contiguous()

        def selfAttention_fp(input, context, input_mask):
            if config.dtype in [torch.half, torch.float16] and input.dtype == torch.float32:
                input = input.half()
            head_size = input.shape[-1] // config.heads
            do_flash_attn = (head_size <= 128)
            scale = (1 / norm_factor) * (1 / norm_factor)
            if do_flash_attn and context is None:
                qkv_out = linear_func(input, attn_qkvw, attn_qkvb if attn_qkvb is not None else attn_qkvw, attn_qkvb
                                      is not None, do_flash_attn, config.heads, False, rope_theta)

                context_layer = triton_flash_attn_kernel(qkv_out[0], qkv_out[1], qkv_out[2], scale,
                                                         input.shape[-2] % 128 == 0)
                context_layer = _transpose_for_context(context_layer[:, :, :, :head_size])

            else:
                do_flash_attn = False
                if context is not None:
                    query = torch.matmul(input, attn_qw)
                    key = torch.matmul(context, attn_kw)
                    value = torch.matmul(context, attn_vw)
                else:
                    qkv = torch.matmul(input, attn_qkvw)
                    query, key, value = qkv.chunk(3, dim=-1)
                    query = query.contiguous()
                    key = key.contiguous()
                    value = value.contiguous()
                query, key, value = pad_transform_func(query, key, value, config.heads, do_flash_attn)
                attention_scores = (torch.matmul(query, key.transpose(-1, -2)) * scale).softmax(dim=-1)
                context_layer = _transpose_for_context(torch.matmul(attention_scores, value))

            output = linear_func(context_layer, attn_ow, attn_ob, do_out_bias, False, config.heads, False, rope_theta)
            return output

        output = selfAttention_fp(input, context, input_mask)

        return output

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError('You are running with DeepScale Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepScaleDiffusersAttention(nn.Module):
    """Initialize the DeepScale Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepScaleInferenceConfig
    """
    layer_id = 0

    def __init__(
        self,
        config,
    ):
        super(DeepScaleDiffusersAttention, self).__init__()

        self.config = config
        self.config.layer_id = DeepScaleDiffusersAttention.layer_id
        DeepScaleDiffusersAttention.layer_id += 1
        device = get_accelerator().current_device_name() if config.bigscience_bloom else 'cpu'
        qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3

        data_type = self.config.dtype
        data_type_fp = torch.half if self.config.dtype == torch.int8 else self.config.dtype

        if DeepScaleDiffusersAttention.layer_id == 1:
            log_dist(f"DeepScale-Attention config: {self.config.__dict__}", [0])

        self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                  qkv_size_per_partition,
                                                  dtype=data_type,
                                                  device=device),
                                      requires_grad=False)
        self.attn_kw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_vw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_qw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device),
                                      requires_grad=False)
        out_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)

        self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
                                    requires_grad=False)
        self.do_out_bias = True

        if triton_flash_attn is None:
            load_triton_flash_attn()
        self.triton_flash_attn_kernel = triton_flash_attn()
        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.norm_factor = math.sqrt(math.sqrt(self.config.hidden_size // self.config.heads))

        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L191

        self.workspace = WorkspaceOp(self.config)
        self.score_context_func = SoftmaxContextOp(self.config)
        self.linear_func = LinearOp(self.config)
        self.pad_transform_func = PadTransformOp(self.config)

    def allocate_workspace(self, size):
        # Allocate memory only on first layer forward
        if self.config.layer_id == 0:
            self.workspace.allocate_workspace(self.config.hidden_size, self.config.heads, size[1], size[0],
                                              DeepScaleDiffusersAttention.layer_id, self.config.mp_size, False, 0,
                                              self.config.max_out_tokens, self.config.min_out_tokens)

    def forward(self, input, context=None, input_mask=None):
        self.allocate_workspace(input.size())
        output = DeepScaleDiffusersAttentionFunction.apply(
            input, context, input_mask, self.config, self.attn_qkvw, self.attn_qw, self.attn_kw, self.attn_vw,
            self.attn_qkvb, self.num_attention_heads_per_partition, self.norm_factor, self.hidden_size_per_partition,
            self.attn_ow, self.attn_ob, self.do_out_bias, self.score_context_func, self.linear_func,
            self.pad_transform_func, self.triton_flash_attn_kernel, self.config.rope_theta)

        return output
