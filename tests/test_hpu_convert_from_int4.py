import torch
import pytest
import habana_frameworks.torch.core as htcore

def test_1d_int():
    out_dtype = torch.bfloat16
    input = torch.full((4,), 0b00010010001101000101011001111000, dtype=torch.int).to("hpu")
    scale = torch.full((4,), 2, dtype=out_dtype).to("hpu")
    zero_point = torch.zeros((4,), dtype=out_dtype).to("hpu")
    result = torch.ops.hpu.convert_from_int4(input, scale, zero_point, out_dtype).cpu().reshape(-1, 8)
    print(result)

def test_1d_uint():
    out_dtype = torch.bfloat16
    input = torch.full((4,), 0b00010010001101000101011001111000, dtype=torch.int).to("hpu")
    scale = torch.full((4,), 2, dtype=out_dtype).to("hpu")
    zero_point = torch.zeros((4,), dtype=out_dtype).to("hpu")
    result = torch.ops.hpu.convert_from_uint4(input, scale, zero_point, out_dtype).cpu().reshape(-1, 8)
    print(result)

@pytest.mark.parametrize("bits", [4])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("infeatures", [4096, 11008])
@pytest.mark.parametrize("outfeatures", [4096, 11008])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("use_cuda_fp16", [False])
@pytest.mark.parametrize("kernel_switch_threshold", [128])
@pytest.mark.parametrize("trainable", [False])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16])
def test_qlinear_hpu(bits, group_size, infeatures, outfeatures, bias, use_cuda_fp16, kernel_switch_threshold, trainable, weight_dtype):
    from auto_gptq.nn_modules.qlinear import qlinear_hpu, qlinear_cuda_old
    quant_hpu = qlinear_hpu.QuantLinear(bits=bits, group_size=group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, use_cuda_fp16=use_cuda_fp16, kernel_switch_threshold=kernel_switch_threshold, trainable=trainable, weight_dtype=weight_dtype).to("hpu")
    # >               zeros = zeros + 1
    # E               RuntimeError: Graph compile failed. synStatus=synStatus 26 [Generic failure].
    # quant_ref = qlinear_cuda_old.QuantLinear(bits=bits, group_size=group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, use_cuda_fp16=use_cuda_fp16, kernel_switch_threshold=kernel_switch_threshold, trainable=trainable, weight_dtype=weight_dtype).to("hpu")
    quant_ref = qlinear_cuda_old.QuantLinear(bits=bits, group_size=group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, use_cuda_fp16=use_cuda_fp16, kernel_switch_threshold=kernel_switch_threshold, trainable=trainable, weight_dtype=weight_dtype)
    # input = torch.rand((infeatures, outfeatures), dtype=torch.float32).to("hpu")
    input = torch.rand((infeatures, outfeatures), dtype=weight_dtype).to("hpu")
    out_hpu = quant_hpu(input)
    out_test = out_hpu.to('cpu')
    out_ref = quant_ref(input)
    # assert torch.allclose(out_hpu, out_ref, rtol=1e-3, atol=1e-3))
    print(f"{out_hpu.dtype=} {out_ref.dtype=} {out_test.dtype=}")
    assert torch.allclose(out_test, out_ref)
    assert torch.allclose(out_hpu, out_ref)