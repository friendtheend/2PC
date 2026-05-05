#!/usr/bin/env python3

# Modified by Andes Y. L. Kei: Implemented PyTorch to ONNX conversion for GELU, Embedding
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import inspect
import io
import os
import shutil
import tempfile

import onnx
import torch
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.utils
from onnx import numpy_helper
from torch.onnx import OperatorExportTypes

from . import module

try:
    import tensorflow as tf  # noqa
    import tf2onnx

    TF_AND_TF2ONNX = True
except ImportError:
    TF_AND_TF2ONNX = False

try:
    import torch.onnx.symbolic_registry as sym_registry  # noqa

    SYM_REGISTRY = True
except ImportError:
    # Newer PyTorch versions removed both `symbolic_registry` and the old
    # `_internal.registration` import path used by legacy CrypTen code.
    # In that case we rely on `torch.onnx.register_custom_op_symbolic` below.
    SYM_REGISTRY = False


_OPSET_VERSION = 17
_ONNX_PROTOBUF_SOFT_LIMIT_BYTES = int(1.5 * (1024 ** 3))


def from_onnx(onnx_string_or_file):
    """
    Converts an ONNX model serialized in an `onnx_string_or_file` to a CrypTen model.
    """
    onnx_model = _load_onnx_model(onnx_string_or_file)
    return _to_crypten(onnx_model)


def from_pytorch(pytorch_model, dummy_input, **kwargs):
    """
    Converts a PyTorch model `pytorch_model` into a CrypTen model by tracing it
    using the input `dummy_input`.
    """

    # construct CrypTen model:
    tmpdir = None
    f = None
    force_disk = os.environ.get("CRYPTEN_ONNX_FORCE_DISK", "0") == "1"
    model_bytes = _model_parameter_bytes(pytorch_model)
    prefer_disk = force_disk or (model_bytes >= _ONNX_PROTOBUF_SOFT_LIMIT_BYTES)
    try:
        if prefer_disk:
            model_path, tmpdir = _from_pytorch_to_disk(pytorch_model, dummy_input, **kwargs)
            crypten_model = _to_crypten(onnx.load(model_path))
        else:
            f = _from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs)
            crypten_model = from_onnx(f)
    except RuntimeError as exc:
        # Large LLMs (e.g., Llama2-7B) exceed protobuf's 2GiB in-memory ONNX
        # limit. In that case export to a file so ONNX external data can be used.
        if not _is_onnx_2gb_export_error(exc):
            raise
        model_path, tmpdir = _from_pytorch_to_disk(pytorch_model, dummy_input, **kwargs)
        crypten_model = _to_crypten(onnx.load(model_path))
    finally:
        if f is not None:
            f.close()
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Keep PyTorch copy only when needed. Deep-copying multi-billion parameter
    # models is very expensive and unnecessary for inference-only benchmarking.
    keep_pytorch_model = os.environ.get("CRYPTEN_KEEP_PYTORCH_MODEL", "0") == "1"
    if keep_pytorch_model:
        crypten_model.pytorch_model = copy.deepcopy(pytorch_model)
    else:
        crypten_model.pytorch_model = None

    # make sure training / eval setting is copied:
    crypten_model.train(mode=pytorch_model.training)
    return crypten_model


def from_tensorflow(tensorflow_graph_def, inputs, outputs):
    """
    Function that converts Tensorflow model into CrypTen model based on
    https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/convert.py
    The model is returned in evaluation mode.
    Args:
        `tensorflow_graph_def`: Input Tensorflow GraphDef to be converted
        `inputs`: input nodes
        `outputs`: output nodes
    """
    raise DeprecationWarning(
        "crypten.nn.from_tensorflow is deprecated. ",
        "CrypTen will no longer support model conversion from TensorFlow.",
    )
    # Exporting model to ONNX graph
    if not TF_AND_TF2ONNX:
        raise ImportError("Please install both tensorflow and tf2onnx packages")

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(tensorflow_graph_def, name="")
    with tf2onnx.tf_loader.tf_session(graph=tf_graph):
        g = tf2onnx.tfonnx.process_tf_graph(
            tf_graph,
            opset=10,
            continue_on_error=False,
            input_names=inputs,
            output_names=outputs,
        )
    onnx_graph = tf2onnx.optimizer.optimize_graph(g)
    model_proto = onnx_graph.make_model(
        "converted from {}".format(tensorflow_graph_def)
    )
    f = io.BytesIO()
    f.write(model_proto.SerializeToString())

    # construct CrypTen model
    # Note: We don't convert crypten model to training mode, as Tensorflow
    # models are used for both training and evaluation without the specific
    # conversion of one mode to another
    f.seek(0)
    crypten_model = from_onnx(f)
    return crypten_model


def _from_pytorch_to_bytes(pytorch_model, dummy_input, **kwargs):
    """
    Returns I/O stream containing ONNX graph for `pytorch_model` traced with
    input `dummy_input`.
    """

    # first export is only used to obtain the PyTorch-to-ONNX symbolic registry:
    with io.BytesIO() as f:
        _export_pytorch_model(f, pytorch_model, dummy_input, **kwargs)

    # update ONNX symbolic registry with CrypTen-specific functions:
    _update_onnx_symbolic_registry()

    # export again so the graph is created with CrypTen-specific registry:
    f = io.BytesIO()
    f = _export_pytorch_model(f, pytorch_model, dummy_input, **kwargs)
    f.seek(0)
    return f


def _from_pytorch_to_disk(pytorch_model, dummy_input, **kwargs):
    """
    Exports ONNX graph for large models to disk so external tensor data can be
    written alongside the graph (avoids protobuf 2GiB in-memory limit).
    Returns `(model_path, tmpdir)`.
    """

    tmpdir = tempfile.mkdtemp(prefix="crypten_onnx_")
    model_path = os.path.join(tmpdir, "model.onnx")
    export_kwargs = dict(kwargs) if kwargs is not None else {}
    # Keep kwargs minimal for broad torch-version compatibility. For large
    # models, exporting to a file path is sufficient to enable external data
    # handling in older torch/onnx stacks.

    # update ONNX symbolic registry with CrypTen-specific functions:
    _update_onnx_symbolic_registry()

    # export graph to a file path so large weights can spill to external data.
    _export_pytorch_model(model_path, pytorch_model, dummy_input, **export_kwargs)
    return model_path, tmpdir


def _is_onnx_2gb_export_error(exc):
    message = str(exc)
    return "larger than the 2GiB limit imposed by the protobuf library" in message


def _model_parameter_bytes(pytorch_model):
    total = 0
    for param in pytorch_model.parameters():
        total += param.numel() * param.element_size()
    return total


def _export_pytorch_model(f, pytorch_model, dummy_input, **kwargs):
    """
    Returns a binary I/O stream containing ONNX-exported pytorch_model that was
    traced with input `dummy_input`.
    """
    default_kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "input_names": ["input"],
        "operator_export_type": OperatorExportTypes.ONNX,
        "output_names": ["output"],
        "opset_version": _OPSET_VERSION,
        "dynamic_axes": {"input" : {0 : "batch_size"}, "output" : {0 : "batch_size"}}
    }
    if kwargs is not None:
        kwargs.update(default_kwargs)
    else:
        kwargs = default_kwargs

    # Different torch versions accept different ONNX export kwargs.
    # Drop unsupported keys to keep conversion compatible across environments.
    supported = set(inspect.signature(torch.onnx.export).parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in supported}
    kwargs.setdefault("verbose", False)

    torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    return f


# mapping from ONNX to crypten.nn for modules with different names:
ONNX_TO_CRYPTEN = {
    "adaptive_avg_pool2d": module.AdaptiveAvgPool2d,
    "adaptive_max_pool2d": module.AdaptiveMaxPool2d,
    "AveragePool": module.AvgPool2d,
    "Clip": module.Hardtanh,
    "MaxPool": module.MaxPool2d,
    "Pad": module._ConstantPad,
    "Relu": module.ReLU,
    "ReduceMean": module.Mean,
    "ReduceSum": module.Sum,
}


def _to_crypten(onnx_model):
    """
    Function that converts an `onnx_model` to a CrypTen model.
    """

    # create graph:
    input_names, output_names = _get_input_output_names(onnx_model)
    assert len(output_names) == 1, "Only one output per model supported."
    crypten_model = module.Graph(input_names, output_names[0])

    # create nodes for the parameters:
    for node in onnx_model.graph.initializer:
        param = torch.from_numpy(numpy_helper.to_array(node))
        crypten_model.add_module(node.name, module.Parameter(param), [])

    # Detect ONNX pattern SiLU(x) = Sigmoid(x) * x and fuse it into one
    # CrypTen SiLU node so cfg.functions.silu_method is actually exercised.
    consumers_by_tensor = {}
    for node in onnx_model.graph.node:
        for in_name in node.input:
            consumers_by_tensor.setdefault(in_name, []).append(node)

    fused_mul_output_to_input = {}
    skipped_sigmoid_outputs = set()
    for node in onnx_model.graph.node:
        if node.op_type != "Sigmoid" or len(node.input) != 1 or len(node.output) != 1:
            continue
        sig_in = node.input[0]
        sig_out = node.output[0]
        consumers = consumers_by_tensor.get(sig_out, [])
        if len(consumers) != 1:
            continue
        mul_node = consumers[0]
        if mul_node.op_type != "Mul" or len(mul_node.input) != 2 or len(mul_node.output) < 1:
            continue
        if sig_in not in mul_node.input:
            continue
        fused_mul_output_to_input[mul_node.output[0]] = sig_in
        skipped_sigmoid_outputs.add(sig_out)

    # loop over all nodes:
    for node in onnx_model.graph.node:
        if node.op_type == "Sigmoid" and len(node.output) == 1 and node.output[0] in skipped_sigmoid_outputs:
            continue

        # get attributes and node type:
        attributes = {attr.name: _get_attribute_value(attr) for attr in node.attribute}
        output_names = list(node.output)
        if node.op_type == "Mul" and output_names and output_names[0] in fused_mul_output_to_input:
            crypten_module = module.SiLU()
            input_names = [fused_mul_output_to_input[output_names[0]]]
            crypten_model.add_module(
                output_names[0], crypten_module, input_names, output_names=output_names
            )
            continue
        crypten_class = _get_operator_class(node.op_type, attributes)

        # add CrypTen module to graph:
        crypten_module = crypten_class.from_onnx(attributes=attributes)
        input_names = list(node.input)
        if node.op_type == "Dropout":
            output_names = [output_names[0]]  # do not output Dropout mask
        crypten_model.add_module(
            output_names[0], crypten_module, input_names, output_names=output_names
        )

    # return final model:
    crypten_model = _get_model_or_module(crypten_model)
    return crypten_model


def _load_onnx_model(onnx_string_or_file):
    """
    Loads ONNX model from file or string.
    """
    if isinstance(onnx_string_or_file, (str, os.PathLike)) and os.path.exists(onnx_string_or_file):
        return onnx.load(onnx_string_or_file)
    if hasattr(onnx_string_or_file, "seek"):
        onnx_string_or_file.seek(0)
        return onnx.load(onnx_string_or_file)
    return onnx.load_model_from_string(onnx_string_or_file)


def _get_input_output_names(onnx_model):
    """
    Return input and output names of the ONNX graph.
    """
    input_names = [input.name for input in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]
    available_outputs = []
    for node in onnx_model.graph.node:
        available_outputs.extend(list(node.output))

    forced_output_name = os.environ.get("CRYPTEN_PROFILE_OUTPUT_NAME", "").strip()
    forced_output_regex = os.environ.get("CRYPTEN_PROFILE_OUTPUT_REGEX", "").strip()
    if forced_output_name:
        if forced_output_name in available_outputs:
            output_names = [forced_output_name]
        else:
            print(
                f"WARNING: CRYPTEN_PROFILE_OUTPUT_NAME '{forced_output_name}' not found in graph outputs; "
                "falling back to default ONNX outputs."
            )
    elif forced_output_regex:
        import re

        pattern = re.compile(forced_output_regex)
        matched = next((name for name in available_outputs if pattern.search(name)), None)
        if matched is not None:
            output_names = [matched]
        else:
            print(
                f"WARNING: CRYPTEN_PROFILE_OUTPUT_REGEX '{forced_output_regex}' matched nothing; "
                "falling back to default ONNX outputs."
            )
    assert len(input_names) >= 1, "number of inputs should be at least 1"
    # TODO: handle multiple outputs
    # assert len(output_names) == 1, "number of outputs should be 1"
    if len(output_names) > 1:
        output_names = [output_names[0]]
    return input_names, output_names


def _get_model_or_module(crypten_model):
    """
    Returns `Module` if model contains only one module. Otherwise returns model.
    """
    num_modules = len(list(crypten_model.modules()))
    if num_modules == 1:
        for crypten_module in crypten_model.modules():
            return crypten_module
    return crypten_model


def _get_attribute_value(attr):
    """
    Retrieves value from an ONNX attribute.
    """
    if attr.HasField("f"):  # floating-point attribute
        return attr.f
    elif attr.HasField("i"):  # integer attribute
        return attr.i
    elif attr.HasField("s"):  # string attribute
        return attr.s  # TODO: Sanitize string.
    elif attr.HasField("t"):  # tensor attribute
        return torch.from_numpy(numpy_helper.to_array(attr.t))
    elif len(attr.ints) > 0:
        return list(attr.ints)
    elif len(attr.floats) > 0:
        return list(attr.floats)
    raise ValueError("Unknown attribute type for attribute %s." % attr.name)


def _get_operator_class(node_op_type, attributes):
    """
    Returns the `crypten.nn.Module` type corresponding to an ONNX node.
    """
    crypten_class = getattr(
        module, node_op_type, ONNX_TO_CRYPTEN.get(node_op_type, None)
    )
    if crypten_class is None:
        raise ValueError(f"CrypTen does not support ONNX op {node_op_type}.")
    return crypten_class


def _update_onnx_symbolic_registry():
    """
    Updates the ONNX symbolic registry for operators that need a CrypTen-specific
    implementation and custom operators.
    """
    if SYM_REGISTRY:
        # update PyTorch's symbolic ONNX registry to output different functions:
        for version_key, version_val in sym_registry._registry.items():
            for function_key in version_val.keys():
                if function_key == "softmax":
                    sym_registry._registry[version_key][
                        function_key
                    ] = _onnx_crypten_softmax
                if function_key == "log_softmax":
                    sym_registry._registry[version_key][
                        function_key
                    ] = _onnx_crypten_logsoftmax
                if function_key == "dropout":
                    sym_registry._registry[version_key][
                        function_key
                    ] = _onnx_crypten_dropout
                if function_key == "feature_dropout":
                    sym_registry._registry[version_key][
                        function_key
                    ] = _onnx_crypten_feature_dropout
                if function_key == "gelu":
                    sym_registry._registry[version_key][
                        function_key
                    ] = _onnx_shaft_gelu
                if function_key == "embedding":
                    sym_registry._registry[version_key][
                        function_key
                    ] = _onnx_shaft_embedding
    else:
        # Update ONNX symbolic registry using torch.onnx.register_custom_op_symbolic
        torch.onnx.register_custom_op_symbolic(
            "aten::softmax", _onnx_crypten_softmax, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::log_softmax", _onnx_crypten_logsoftmax, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::dropout", _onnx_crypten_dropout, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::feature_dropout", _onnx_crypten_feature_dropout, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::gelu", _onnx_shaft_gelu, _OPSET_VERSION
        )
        torch.onnx.register_custom_op_symbolic(
            "aten::embedding", _onnx_shaft_embedding, _OPSET_VERSION
        )


@sym_help.parse_args("v", "i", "none")
def _onnx_crypten_softmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's Softmax module to a Softmax module in
    the ONNX model. It overrides PyTorch's default conversion of Softmax module
    to a sequence of Exp, ReduceSum and Div modules, since this default
    conversion can cause numerical overflow when applied to CrypTensors.
    """
    result = g.op("Softmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "i", "none")
def _onnx_crypten_logsoftmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's LogSoftmax module to a LogSoftmax module in
    the ONNX model. It overrides PyTorch's default conversion of LogSoftmax module
    to avoid potentially creating Transpose operators.
    """
    result = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "f", "i")
def _onnx_crypten_dropout(g, input, p, train):
    """
    This function converts PyTorch's Dropout module to a Dropout module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the Dropout module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore Dropout modules are not required in the ONNX model.
    However, CrypTen needs to convert ONNX models to trainable
    CrypTen models, and so the Dropout module needs to be included in the
    CrypTen-specific conversion.
    """
    ratio = g.op("Constant", value_t=torch.tensor(p))
    training_mode = g.op("Constant", value_t=torch.tensor(train))
    r, _ = g.op("Dropout", input, ratio, training_mode, outputs=2)
    return r


@sym_help.parse_args("v", "f", "i")
def _onnx_crypten_feature_dropout(g, input, p, train):
    """
    This function converts PyTorch's DropoutNd module to a DropoutNd module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the DropoutNd module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore DropoutNd modules are not required in the ONNX model.
    However, CrypTen needs to convert ONNX models to trainable
    CrypTen models, and so the DropoutNd module needs to be included in the
    CrypTen-specific conversion.
    """
    # TODO: the code below may be problematic
    ratio = g.op("Constant", value_t=torch.tensor(p))
    r, _ = g.op("DropoutNd", input, ratio, outputs=2)
    return r

@sym_help.parse_args("v", "s")
def _onnx_shaft_gelu(g, input, approximate):
    """
    This function converts PyTorch's GELU module to a GELU module in
    the ONNX model. It overrides PyTorch's default conversion of GELU module.
    """
    return g.op("shaft::GELU", input, approximate_s=approximate).setType(input.type())

'''
Refer to https://github.com/pytorch/pytorch/blob/10c646295d3512237cfb3ab44aa21dfcc9832441/torch/_decomp/decompositions.py#L1164
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    pass
'''
@sym_help.parse_args("v", "v", "v", "v", "v")
def _onnx_shaft_embedding(g, input, input1, input2, input3, input4):
    """
    This function converts PyTorch's Embedding module to a Embedding module in
    the ONNX model. It overrides PyTorch's default conversion of Embedding module.
    """
    return g.op("shaft::Embedding", input, input1, input2)
