#!/usr/bin/env python
# coding=utf-8
# Modified by SHAFT's team: Private Text Generation.
#
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)"""

import argparse
import inspect
import logging
import os
import sys
from typing import Tuple

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAFT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if SHAFT_ROOT not in sys.path:
    sys.path.insert(0, SHAFT_ROOT)

import crypten as ct
from crypten.config import cfg
from multiprocess_launcher import MultiProcessLauncher

from transformers import (
    AutoConfig,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GenerationMixin,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTJForCausalLM,
    LlamaForCausalLM,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    OPTForCausalLM,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    GPTNeoForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
    "gptj": (GPTJForCausalLM, AutoTokenizer),
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    # Use AutoTokenizer for better compatibility across Llama checkpoints.
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "opt": (OPTForCausalLM, GPT2Tokenizer),
    "gpt-neo": (GPTNeoForCausalLM, GPT2Tokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def sparse_model_config(model_config):
    embedding_size = None
    if hasattr(model_config, "hidden_size"):
        embedding_size = model_config.hidden_size
    elif hasattr(model_config, "n_embed"):
        embedding_size = model_config.n_embed
    elif hasattr(model_config, "n_embd"):
        embedding_size = model_config.n_embd

    num_head = None
    if hasattr(model_config, "num_attention_heads"):
        num_head = model_config.num_attention_heads
    elif hasattr(model_config, "n_head"):
        num_head = model_config.n_head

    if embedding_size is None or num_head is None or num_head == 0:
        raise ValueError("Check the model config")

    num_embedding_size_per_head = int(embedding_size / num_head)
    if hasattr(model_config, "n_layer"):
        num_layer = model_config.n_layer
    elif hasattr(model_config, "num_hidden_layers"):
        num_layer = model_config.num_hidden_layers
    else:
        raise ValueError("Number of hidden layers couldn't be determined from the model config")

    return num_layer, num_head, num_embedding_size_per_head


def generate_past_key_values(model, batch_size, seq_len):
    num_block_layers, num_attention_heads, num_embedding_size_per_head = sparse_model_config(model.config)
    if model.config.model_type == "bloom":
        past_key_values = tuple(
            (
                torch.empty(int(num_attention_heads * batch_size), num_embedding_size_per_head, seq_len)
                .to(model.dtype)
                .to(model.device),
                torch.empty(int(num_attention_heads * batch_size), seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    else:
        past_key_values = tuple(
            (
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
                torch.empty(batch_size, num_attention_heads, seq_len, num_embedding_size_per_head)
                .to(model.dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    return past_key_values


def prepare_jit_inputs(inputs, model, tokenizer):
    batch_size = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt")
    dummy_input = dummy_input.to(model.device)
    if model.config.use_cache:
        dummy_input["past_key_values"] = generate_past_key_values(model, batch_size, 1)
    dummy_input["attention_mask"] = torch.cat(
        [
            torch.zeros(dummy_input["attention_mask"].shape[0], 1)
            .to(dummy_input["attention_mask"].dtype)
            .to(model.device),
            dummy_input["attention_mask"],
        ],
        -1,
    )
    return dummy_input


class _ModelFallbackWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        if kwargs["past_key_values"] is None and self._default.config.use_cache:
            kwargs["past_key_values"] = generate_past_key_values(self._default, kwargs["input_ids"].shape[0], 0)
        kwargs.pop("position_ids", None)
        for k in list(kwargs.keys()):
            if kwargs[k] is None or isinstance(kwargs[k], bool):
                kwargs.pop(k)
        outputs = self._optimized(**kwargs)
        lm_logits = outputs[0]
        past_key_values = outputs[1]
        fixed_output = CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        return fixed_output

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--len_data",
        type=int,
        default=128,
        help="Sequence length of data to run.",
    )
    parser.add_argument(
        "--comp",
        action="store_true",
        help="If passed, estimate computation time (without communication).",
    )
    parser.add_argument(
        "--estimate_mode",
        type=str,
        choices=["auto", "comm", "total"],
        default="auto",
        help="Cost print mode for communication run. 'total' prints latency + communication.",
    )
    parser.add_argument(
        "--report_cost",
        action="store_true",
        help="Print detailed per-module latency/communication/round breakdown.",
    )
    parser.add_argument(
        "--gelu_method",
        type=str,
        default=os.environ.get("GELU_METHOD", cfg.functions.gelu_method),
        choices=["ideal", "fourier", "secformer", "poly", "bolt", "erf", "d2poly"],
        help="GELU approximation used by SHAFT crypten backend.",
    )
    parser.add_argument(
        "--silu_method",
        type=str,
        default=os.environ.get("SILU_METHOD", cfg.functions.silu_method),
        choices=["ideal", "fourier", "d2poly"],
        help="SiLU approximation used by SHAFT crypten backend.",
    )
    parser.add_argument(
        "--silu_d2poly_a",
        type=float,
        default=float(os.environ["SILU_D2POLY_A"]) if "SILU_D2POLY_A" in os.environ else None,
        help="Override SiLU d2poly coefficient a.",
    )
    parser.add_argument(
        "--silu_d2poly_b",
        type=float,
        default=float(os.environ["SILU_D2POLY_B"]) if "SILU_D2POLY_B" in os.environ else None,
        help="Override SiLU d2poly coefficient b.",
    )
    parser.add_argument(
        "--silu_d2poly_c",
        type=float,
        default=float(os.environ["SILU_D2POLY_C"]) if "SILU_D2POLY_C" in os.environ else None,
        help="Override SiLU d2poly coefficient c.",
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Whether or not to use cpu. If set to False, " "we will use gpu/npu or mps device if available",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--max_layers",
        type=int,
        default=int(os.environ.get("LLAMA_MAX_LAYERS", "0")),
        help="If > 0 and model_type=llama, keep only the first N decoder layers (for memory-limited profiling).",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Initialize model from config instead of loading pretrained weights (profiling only).",
    )
    parser.add_argument(
        "--profile_hidden_size",
        type=int,
        default=int(os.environ.get("LLAMA_PROFILE_HIDDEN_SIZE", "0")),
        help="Override hidden_size when --random_init is set.",
    )
    parser.add_argument(
        "--profile_intermediate_size",
        type=int,
        default=int(os.environ.get("LLAMA_PROFILE_INTERMEDIATE_SIZE", "0")),
        help="Override intermediate_size when --random_init is set.",
    )
    parser.add_argument(
        "--profile_num_heads",
        type=int,
        default=int(os.environ.get("LLAMA_PROFILE_NUM_HEADS", "0")),
        help="Override num_attention_heads when --random_init is set.",
    )
    parser.add_argument(
        "--profile_num_kv_heads",
        type=int,
        default=int(os.environ.get("LLAMA_PROFILE_NUM_KV_HEADS", "0")),
        help="Override num_key_value_heads when --random_init is set.",
    )
    parser.add_argument("--jit", action="store_true", help="Whether or not to use jit trace to accelerate inference")
    args = parser.parse_args()
    return args


def _resolve_runtime_device(args):
    if args.use_cpu or not torch.cuda.is_available():
        return "cpu"

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        return "cpu"

    # Optional override for rank->GPU assignment.
    # Example: SHAFT_CUDA_DEVICES=0,1 (logical IDs within visible GPUs).
    device_map = os.environ.get("SHAFT_CUDA_DEVICES", "").strip()
    if device_map:
        try:
            candidate_ids = [int(x.strip()) for x in device_map.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid SHAFT_CUDA_DEVICES='{device_map}'. Expected comma-separated integers.") from exc
        if not candidate_ids:
            raise ValueError("SHAFT_CUDA_DEVICES is set but empty after parsing.")
        for idx in candidate_ids:
            if idx < 0 or idx >= visible_gpu_count:
                raise ValueError(
                    f"Invalid SHAFT_CUDA_DEVICES index {idx}; visible GPU range is [0, {visible_gpu_count - 1}]."
                )
    else:
        candidate_ids = list(range(visible_gpu_count))

    if world_size > len(candidate_ids):
        logger.warning(
            "WORLD_SIZE=%s but only %s GPU slot(s) configured; some ranks will share GPUs and may OOM.",
            world_size,
            len(candidate_ids),
        )

    device_id = candidate_ids[rank % len(candidate_ids)]
    torch.cuda.set_device(device_id)
    return f"cuda:{device_id}"


def main():
    args = parse_args()
    device = _resolve_runtime_device(args)
    cfg.debug.report_cost = bool(args.report_cost)
    cfg.functions.gelu_method = args.gelu_method
    cfg.functions.silu_method = args.silu_method
    if args.silu_d2poly_a is not None:
        cfg.functions.silu_d2poly_a = args.silu_d2poly_a
    if args.silu_d2poly_b is not None:
        cfg.functions.silu_d2poly_b = args.silu_d2poly_b
    if args.silu_d2poly_c is not None:
        cfg.functions.silu_d2poly_c = args.silu_d2poly_c

    if args.comp:
        cfg.cost.estimate_cost = True
        cfg.cost.estimate_mode = "comp"
    elif args.estimate_mode != "auto":
        cfg.cost.estimate_cost = True
        cfg.cost.estimate_mode = args.estimate_mode

    logger.warning(
        "device: %s, rank: %s, world_size: %s, 16-bits inference: %s",
        device,
        os.environ.get("RANK", "0"),
        os.environ.get("WORLD_SIZE", "1"),
        args.fp16,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.random_init:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        if args.profile_hidden_size > 0:
            config.hidden_size = args.profile_hidden_size
        if args.profile_intermediate_size > 0:
            config.intermediate_size = args.profile_intermediate_size
        if args.profile_num_heads > 0:
            config.num_attention_heads = args.profile_num_heads
        if args.profile_num_kv_heads > 0 and hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = args.profile_num_kv_heads
        if args.max_layers > 0 and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = args.max_layers
        if hasattr(config, "hidden_size") and hasattr(config, "num_attention_heads"):
            assert config.hidden_size % config.num_attention_heads == 0, (
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        model = model_class(config)
        if args.fp16:
            model = model.half()
        logger.warning(
            "Using random-init model for profiling: layers=%s hidden=%s intermediate=%s heads=%s kv_heads=%s",
            getattr(config, "num_hidden_layers", "n/a"),
            getattr(config, "hidden_size", "n/a"),
            getattr(config, "intermediate_size", "n/a"),
            getattr(config, "num_attention_heads", "n/a"),
            getattr(config, "num_key_value_heads", "n/a"),
        )
    else:
        model_kwargs = {}
        if args.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        model = model_class.from_pretrained(args.model_name_or_path, **model_kwargs)

    if (
        args.model_type == "llama"
        and args.max_layers > 0
        and hasattr(model, "model")
        and hasattr(model.model, "layers")
    ):
        total_layers = len(model.model.layers)
        if args.max_layers < total_layers:
            model.model.layers = torch.nn.ModuleList(list(model.model.layers[: args.max_layers]))
            if hasattr(model.config, "num_hidden_layers"):
                model.config.num_hidden_layers = args.max_layers
            logger.warning("Truncated Llama layers: %s -> %s", total_layers, args.max_layers)

    # Ensure parameters are fp16 before moving to GPU, reducing peak GPU memory.
    if args.fp16 and next(model.parameters()).dtype != torch.float16:
        model.half()

    # Set the model to the right device
    model.to(device)
    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.length = adjust_length_to_model(args.length, max_sequence_length=max_seq_length)
    logger.info(args)

    if args.prompt:
        prompt_text = args.prompt
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = args.prefix if args.prefix else args.padding_text
            encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
    else:
        token_id = tokenizer.eos_token_id
        if token_id is None:
            token_id = tokenizer.pad_token_id
        if token_id is None:
            token_id = 0
        encoded_prompt = torch.full((1, args.len_data), token_id, dtype=torch.long)
    encoded_prompt = encoded_prompt.to(device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    if args.jit:
        jit_input_texts = ["enable jit"]
        jit_inputs = prepare_jit_inputs(jit_input_texts, model, tokenizer)
        torch._C._jit_set_texpr_fuser_enabled(False)
        model.config.return_dict = False
        if hasattr(model, "forward"):
            sig = inspect.signature(model.forward)
        else:
            sig = inspect.signature(model.__call__)
        jit_inputs = tuple(jit_inputs[key] for key in sig.parameters if jit_inputs.get(key, None) is not None)
        traced_model = torch.jit.trace(model, jit_inputs, strict=False)
        traced_model = torch.jit.freeze(traced_model.eval())
        traced_model(*jit_inputs)
        traced_model(*jit_inputs)

        model = _ModelFallbackWrapper(traced_model, model)

    # GPT2-specific activation patch. Llama and other families do not expose
    # model.transformer.h with the same structure.
    if args.model_type == "gpt2" and hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for block in model.transformer.h:
            if hasattr(block, "mlp") and hasattr(block.mlp, "act"):
                block.mlp.act = torch.nn.GELU(approximate="tanh")

    ct.init()
    model_enc = ct.nn.from_pytorch(model, model.dummy_inputs["input_ids"].to(device))
    # Reduce peak GPU memory: once ONNX->CrypTen conversion is done, the
    # original PyTorch model is no longer needed for this benchmark path.
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    model_enc = model_enc.encrypt().to(device)
    if os.environ.get("CRYPTEN_PUBLIC_INPUT", "0") == "1":
        input_feed = input_ids.to(device)
    else:
        input_feed = ct.cryptensor(input_ids).to(device)
    model_enc(input_feed)
    return


if __name__ == "__main__":
    args = parse_args()
    if args.comp:
        # run without communication
        main()
    else:
        # run with communication
        launcher = MultiProcessLauncher(int(os.environ.get("WORLD_SIZE", "1")), main)
        launcher.start()
        launcher.join()
        launcher.terminate()
