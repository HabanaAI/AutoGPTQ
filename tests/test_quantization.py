import os
import tempfile
import unittest

import torch.cuda
from parameterized import parameterized
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.quantization import CHECKPOINT_FORMAT, QUANT_CONFIG_FILENAME, BaseQuantizeConfig

try:
    import habana_frameworks.torch.core as htcore
    HPU_AVAILABLE = True
except Exception as e:
    HPU_AVAILABLE = False

class TestQuantization(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_quantize(self, use_marlin: bool):
        if HPU_AVAILABLE and use_marlin:
            return unittest.skip(reason="HPU does not support marlin.")

        pretrained_model_dir = "saibo/llama-1B"

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
        examples = [
            tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            ),
            tokenizer(
                "Today I am in Paris and it is a wonderful day."
            ),
        ]

        quantize_config = BaseQuantizeConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            checkpoint_format=CHECKPOINT_FORMAT.MARLIN if use_marlin else CHECKPOINT_FORMAT.GPTQ,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            pretrained_model_dir,
            quantize_config=quantize_config,
            use_flash_attention_2=False,
        )

        model.quantize(examples)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            device = None if HPU_AVAILABLE else "cuda:0"
            model = AutoGPTQForCausalLM.from_quantized(tmpdirname, device=device, use_marlin=use_marlin)
            del model
            if not HPU_AVAILABLE:
                torch.cuda.empty_cache()

            # test compat: 1) with simple dict type 2) is_marlin_format
            compat_quantize_config = {
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "is_marlin_format": use_marlin,
            }
            model = AutoGPTQForCausalLM.from_quantized(tmpdirname, device=device, quantize_config=compat_quantize_config)
            assert(isinstance(model.quantize_config, BaseQuantizeConfig))

            del model
            if not HPU_AVAILABLE:
                torch.cuda.empty_cache()

            # test checkinpoint_format hint to from_quantized()
            os.remove(f"{tmpdirname}/{QUANT_CONFIG_FILENAME}")

            compat_quantize_config = {
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
            }
            model = AutoGPTQForCausalLM.from_quantized(tmpdirname, device=device,
                    quantize_config=compat_quantize_config,
                    checkpoint_format=CHECKPOINT_FORMAT.MARLIN if use_marlin else None)
            assert (isinstance(model.quantize_config, BaseQuantizeConfig))
