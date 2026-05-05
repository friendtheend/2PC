# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_vit -- --config `pwd`/examples/python/conf/2pc.json

import argparse
import json
import os
import time
from contextlib import contextmanager

import flax.linen as fnn
import jax
import jax.nn as jnn
import requests
from PIL import Image

# Reference: https://huggingface.co/docs/transformers/model_doc/vit#transformers.FlaxViTForImageClassification
from transformers import AutoImageProcessor, FlaxViTForImageClassification

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
parser.add_argument("--num-images", type=int, default=3)
parser.add_argument("--spu-only", action="store_true")
parser.add_argument(
    "--gelu-mode",
    type=str,
    default=os.environ.get("GELU_MODE", "seg3_gelu"),
    choices=["seg3_gelu", "native"],
)
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def _seg3_gelu(x):
    return intrinsic.spu_vit_gelu(x)


def _softmax(x, axis=-1, where=None, initial=None):
    x_max = jax.numpy.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # exp on large negative is clipped to zero
    nexp = intrinsic.spu_neg_exp(x)
    divisor = jax.numpy.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


def _resolve_gelu(mode):
    return _seg3_gelu if mode == "seg3_gelu" else jnn.gelu


@contextmanager
def patch_vit_act2fn(gelu_fn):
    # FlaxViTIntermediate binds activation from ACT2FN at model construction time.
    from transformers.models.vit import modeling_flax_vit as vit_model

    old_vit_gelu = vit_model.ACT2FN.get("gelu", None)
    vit_model.ACT2FN["gelu"] = gelu_fn
    try:
        yield
    finally:
        if old_vit_gelu is not None:
            vit_model.ACT2FN["gelu"] = old_vit_gelu


@contextmanager
def hijack(gelu_fn, enabled=True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_gelu = jnn.gelu
    fnn_gelu = fnn.gelu
    jnn_sm = jnn.softmax
    fnn_sm = fnn.softmax

    jnn.gelu = gelu_fn
    fnn.gelu = gelu_fn
    jnn.softmax = _softmax
    fnn.softmax = _softmax

    yield
    # recover back
    jnn.gelu = jnn_gelu
    fnn.gelu = fnn_gelu
    jnn.softmax = jnn_sm
    fnn.softmax = fnn_sm


def run_on_cpu(model, inputs):
    print(f"Running on CPU ...")
    params = model.params

    def eval(params, inputs):
        outputs = model(inputs, params=params)
        return outputs.logits

    start = time.time()
    logits = eval(params, inputs)
    end = time.time()
    predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
    print(f"CPU runtime: {(end - start)}s")
    print("Top 5 logits ", logits[:, :5])
    print("CPU Predicted class:", model.config.id2label[predicted_class_idx.item()])

def init_spu(model):
    params = model.params
    return ppd.device("P2")(lambda x: x)(params)
    
def run_on_spu(model, params, inputs, gelu_fn):
    print(f"Running on SPU ...")

    def eval(params, inputs):
        # Patch ACT2FN only during SPU tracing/execution to avoid invoking
        # SPU-only custom calls during model load / CPU init path.
        with patch_vit_act2fn(gelu_fn):
            with hijack(gelu_fn=gelu_fn, enabled=True):
                outputs = model(inputs, params=params)
        return outputs.logits

    inputs = ppd.device("P1")(lambda x: x)(inputs)

    start = time.time()
    logits_spu = ppd.device("SPU")(eval, copts=copts)(params, inputs)
    end = time.time()
    predicted_class_idx = jax.numpy.argmax(ppd.get(logits_spu), axis=-1)
    print(f"SPU runtime: {(end - start)}s")
    print("Top 5 logits ", ppd.get(logits_spu)[:, :5])
    print("SPU Predicted class:", model.config.id2label[predicted_class_idx.item()])
    return end - start


def main():
    gelu_fn = _resolve_gelu(args.gelu_mode)
    print(f"GELU mode: {args.gelu_mode}")
    # Keep model loading on native ops; patching happens only for SPU eval.
    model = FlaxViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    # Keep CPU baseline path native if requested.
    model_cpu = model
    if not args.spu_only and args.gelu_mode != "native":
        model_cpu = FlaxViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    # Init SPU for multiple runs
    params = init_spu(model)
    # the pre-processor is supposed to be public
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    image_names = ["n01440764_tench.JPEG", "n01531178_goldfinch.JPEG", "n01737021_water_snake.JPEG"]
    image_names = image_names[: max(1, args.num_images)]
    spu_runtimes = []
    for name in image_names:
        # load dataset
        image = Image.open("examples/python/imagenet/{}".format(name))
        inputs = image_processor(images=image, return_tensors="np")["pixel_values"]
        print("Name = {} ...".format(name))
        if not args.spu_only:
            run_on_cpu(model_cpu, inputs)
        spu_runtimes.append(run_on_spu(model, params, inputs, gelu_fn))

    avg_runtime = sum(spu_runtimes) / len(spu_runtimes)
    print(
        f"BENCH_RESULT system=OpenBumbleBee model=ViT-B16 latency_sec={avg_runtime:.6f} "
        f"num_images={len(spu_runtimes)}"
    )


if __name__ == "__main__":
    main()
