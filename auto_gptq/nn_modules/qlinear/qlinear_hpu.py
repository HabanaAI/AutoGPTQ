import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers
import habana_frameworks.torch.core as htcore


logger = getLogger(__name__)


class QuantLinear(nn.Module):
    QUANT_TYPE = "hpu"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        use_cuda_fp16=True,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        logger.debug(f"qlinear_hpu QuantLinear::__init__ {bits=}, {group_size=}, {infeatures=}, {outfeatures=}, {bias=}, {use_cuda_fp16=}, {kernel_switch_threshold=}, {trainable=}, {weight_dtype=}")
        super().__init__()
        if bits != 4:
            raise NotImplementedError("Only 4 bits are supported.")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        # is performed by unpacking the weights and using torch.matmul
        self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)

        self.trainable = trainable

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, g_idx):
        print(f"pack {linear=}, {scales=}, {zeros=}, {g_idx=}")
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        intweight = []
        for idx in range(self.infeatures):
            g_idx = idx // self.group_size
            intweight.append(torch.round((W[:, idx] + scale_zeros[g_idx]) / self.scales[g_idx]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        print(f"hpu QuantLinear before setting qweight")
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        # qzeros = qzeros.astype(np.float32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        x_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.outfeatures,)
        print(f"forward before x reshape {x.shape=}")
        x = x.reshape(-1, x.shape[-1])
        print(f"forward after x reshape {x.shape=}")
        # if (
        #     x.device.type == "cuda"
        #     and self.autogptq_cuda_available is True
        #     and (self.kernel_switch_threshold is False or x.shape[0] < self.kernel_switch_threshold)
        # ):
        print(f"QuantLinear::forward {self.wf.device=} {self.qzeros.device=}")
        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)
        print(f"QuantLinear::forward {self.wf.device=} {self.qzeros.device=}")


        print(f"forward {self.qweight.shape=} {self.scales.shape=} {self.qzeros.shape=} {x_dtype=}")
        print(f"forward {self.qweight.dtype=} {self.scales.dtype=} {self.qzeros.dtype=} {x_dtype=}")
        # weight = torch.ops.hpu.convert_from_int4(self.qweight, self.scales, self.qzeros, x_dtype)
        if False:
            self.qzeros = self.qzeros.to(torch.float32)
        qweight = self.qweight
        qzeros = self.qzeros
        scales = self.scales
        reshape_weight = False #True
        reshape_scales_zeros = False #True
        zeros_as_none = True
        if reshape_weight:
            #RuntimeError: Common dimension sizes of matmul inputs should be the same. Got 4096 and 32768
            qweight = qweight.reshape(qweight.shape[-1], -1).contiguous()
            print(f"forward reshape_weight {qweight.shape=} {qweight.dtype=}")
        if reshape_scales_zeros:
            #RuntimeError: Graph compile failed.
            # [13:29:57.521604][HABANA_NODE           ][error][tid:83EA] Output tensor and input tensor of model/0/q_proj/dequantize_4_bit_bf16/155_complex/reshape_2 doesn't match in elements' count ( 0 , 16777216 )
            # [13:29:57.521620][HABANA_NODE           ][error][tid:83EA] Node Validation Failed. Cannot create node model/0/q_proj/dequantize_4_bit_bf16/155_complex/reshape_2.
            # [13:29:57.521772][PASS_MANAGER          ][error][tid:83EA] Graph optimization failed! Got SynapseException: Invalid Node Params! Node name: model/0/q_proj/dequantize_4_bit_bf16/155_complex/reshape_2
            qzeros = qzeros.reshape(qzeros.shape[-1], -1).contiguous()
            scales = scales.reshape(scales.shape[-1], -1).contiguous()
            # qzeros = qzeros.reshape(-1, qzeros.shape[-1]).contiguous()
            # scales = scales.reshape(-1, scales.shape[-1]).contiguous()
            print(f"forward reshape_scales_zeros {scales.shape=} {qzeros.shape=}")
        # weight = torch.ops.hpu.convert_from_uint4(self.qweight, self.scales, self.qzeros, x_dtype)
        # weight = torch.ops.hpu.convert_from_int4(self.qweight, self.scales, self.qzeros, x_dtype)
        if zeros_as_none:
            weight = torch.ops.hpu.convert_from_uint4(qweight, scales, None, x_dtype)
        else:
            weight = torch.ops.hpu.convert_from_uint4(qweight, scales, qzeros, x_dtype)
        print(f"QuantLinear::forward {x.shape=} {weight.shape=} {weight.shape[-1]//8=}")
        # print(f"QuantLinear::forward {x.dtype=} {qweight.dtype=} {weight.shape[-1]//8=}")
        weight = weight.reshape(-1, weight.shape[-1]//8)
        print(f"QuantLinear::forward {weight.shape=}")
        # weight = weight.reshape(-1, self.group_size, weight.shape[2])
        print(f"hpu QuantLinear forward {x.dtype=} {weight.dtype=}")
        out = torch.matmul(x, weight)
        print(f"hpu QuantLinear forward {out.shape=} {out_shape=}")
        # if self.bits not in [4]:
        if True:
            out = out.to(dtype=x_dtype).reshape(
                out_shape
            )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        print(f"{out.shape=} {self.bias.shape=}")
        out = out + self.bias if self.bias is not None else out
        return out


__all__ = ["QuantLinear"]
