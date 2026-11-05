#!/usr/bin/env python3
"""Write tiny, deterministic ONNX models with the same inputs / outputs as the real iSurgical models.

    python3 -I test/tools/make_fake_models.py <output folder>

No `onnx` package is needed: the protobuf bytes are encoded by hand with the standard library.
The models are small enough (a few KB) to run every test in milliseconds on CPU, and their outputs are
known exactly, so the tests can check the C++ code (argmax, softmax, state carrying, post-processing)
instead of the network weights.

Segmentation model  (seg.onnx, same I/O as lc_seg_model.onnx)
    image [1,512,512,3] RGB in [0,1] -> Transpose -> 1x1 Conv per head:
      head_0 (5 ch): red   -> class 1, green -> class 2, blue -> class 3   (Gallbladder / Cystic duct / Cystic artery)
      head_1 (3 ch): yellow (red + green) -> class 1                        (Liver)
      head_2 (4 ch), head_3 (7 ch): always Background
      correlation_scores [1,7] = 0.1, 0.2, ... 0.7
RSD preprocessing   (rsd_pre.onnx, same I/O as preprocess_step.onnx)
    image = frame / 255;  hist_stats_out = hist_stats + 1;  hist_count_out = hist_count + 1
RSD model           (rsd.onnx, same I/O as multimodal_rsd_step.onnx)
    phase_probs = fixed, argmax 2 (0.7);  pred_rsd = 60 * mean(conv_state) + 60;  pred_progress = mean(image)
    z_t = i / 512;  conv_state_out = conv_state + 1;  ssm_state_out = ssm_state
Broken variants (wrong names / missing outputs) are written as well, to test the error paths.
"""

# Pack floats / ints into little-endian bytes
import struct
# Command-line argument (output folder)
import sys
# Create the output folder
import os

# ---------------------------------------------------------------------------
# Minimal protobuf encoder
# ---------------------------------------------------------------------------

# ONNX TensorProto.DataType values
FLOAT, UINT8, INT64 = 1, 2, 7
# ONNX AttributeProto.AttributeType values
A_FLOAT, A_INT, A_TENSOR, A_INTS = 1, 2, 4, 7


def varint(value):
    """Protobuf base-128 varint; negative numbers use 64-bit two's complement."""
    # Negative int64 → unsigned 64-bit
    if value < 0:
        # Two's complement
        value += 1 << 64
    # Output bytes
    out = bytearray()
    # 7 bits per byte, high bit = "more follows"
    while True:
        # Low 7 bits
        byte = value & 0x7F
        # Shift them out
        value >>= 7
        # More to come
        if value:
            # Set continuation bit
            out.append(byte | 0x80)
        # Last byte
        else:
            # No continuation bit
            out.append(byte)
            # Done
            return bytes(out)


def f_varint(field, value):
    """Field with wire type 0 (varint)."""
    # Key = field number << 3 | wire type
    return varint(field << 3 | 0) + varint(value)


def f_bytes(field, data):
    """Field with wire type 2 (length-delimited: bytes, strings, sub-messages)."""
    # Key, length, payload
    return varint(field << 3 | 2) + varint(len(data)) + data


def f_str(field, text):
    """String field (UTF-8)."""
    # Strings are length-delimited bytes
    return f_bytes(field, text.encode())


def f_float(field, value):
    """Field with wire type 5 (32-bit float)."""
    # Key + 4 bytes
    return varint(field << 3 | 5) + struct.pack("<f", value)


# ---------------------------------------------------------------------------
# ONNX messages
# ---------------------------------------------------------------------------

def tensor(name, dims, dtype, values):
    """TensorProto with raw_data."""
    # Pack the values according to the type
    if dtype == FLOAT:
        # float32
        raw = struct.pack("<%df" % len(values), *values)
    # int64
    elif dtype == INT64:
        # int64
        raw = struct.pack("<%dq" % len(values), *values)
    # uint8
    else:
        # bytes
        raw = bytes(values)
    # dims (1), data_type (2), name (8), raw_data (9)
    return b"".join(f_varint(1, d) for d in dims) + f_varint(2, dtype) + f_str(8, name) + f_bytes(9, raw)


def value_info(name, elem_type, shape=None):
    """ValueInfoProto: a graph input or output; shape None = unknown rank."""
    # TypeProto.Tensor: elem_type (1), shape (2) with one Dimension (1) per axis, dim_value (1)
    tensor_type = f_varint(1, elem_type)
    # Known shape
    if shape is not None:
        # TensorShapeProto
        tensor_type += f_bytes(2, b"".join(f_bytes(1, f_varint(1, d)) for d in shape))
    # ValueInfoProto: name (1), type (2) = TypeProto { tensor_type (1) }
    return f_str(1, name) + f_bytes(2, f_bytes(1, tensor_type))


def attr_int(name, value):
    """Integer attribute."""
    # name (1), i (3), type (20)
    return f_str(1, name) + f_varint(3, value) + f_varint(20, A_INT)


def attr_ints(name, values):
    """List-of-integers attribute."""
    # name (1), ints (8, repeated), type (20)
    return f_str(1, name) + b"".join(f_varint(8, v) for v in values) + f_varint(20, A_INTS)


def attr_tensor(name, t):
    """Tensor attribute (used by Constant)."""
    # name (1), t (5), type (20)
    return f_str(1, name) + f_bytes(5, t) + f_varint(20, A_TENSOR)


def node(op, inputs, outputs, attrs=()):
    """NodeProto."""
    # input (1), output (2), name (3), op_type (4), attribute (5)
    return (b"".join(f_str(1, i) for i in inputs) + b"".join(f_str(2, o) for o in outputs) +
            f_str(3, op + "_" + outputs[0]) + f_str(4, op) + b"".join(f_bytes(5, a) for a in attrs))


def model(nodes, inputs, outputs, initializers=()):
    """ModelProto with IR version 8 and opset 17 (both supported by ONNX Runtime 1.20)."""
    # GraphProto: node (1), name (2), initializer (5), input (11), output (12)
    graph = (b"".join(f_bytes(1, n) for n in nodes) + f_str(2, "fake") +
             b"".join(f_bytes(5, t) for t in initializers) +
             b"".join(f_bytes(11, i) for i in inputs) + b"".join(f_bytes(12, o) for o in outputs))
    # ModelProto: ir_version (1), producer_name (2), graph (7), opset_import (8) = { version (2) }
    return f_varint(1, 8) + f_str(2, "isurgical-tests") + f_bytes(7, graph) + f_bytes(8, f_varint(2, 17))


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------

# Channels per head, as in SegInferenceEngine::kHeadClasses
HEAD_CLASSES = [5, 3, 4, 7]


def head_weights(head):
    """1x1 Conv weights [C,3] and bias [C] for one head (input channels are R, G, B in [0, 1])."""
    # Number of channels
    c = HEAD_CLASSES[head]
    # Start with all zeros
    w = [[0.0, 0.0, 0.0] for _ in range(c)]
    # Background logit 0.5, every other class -10 (never wins) unless set below
    b = [0.5] + [-10.0] * (c - 1)
    # head_0: R, G, B each select one class: logit = 4 * colour - 1 (pure colour → 3, black → -1)
    if head == 0:
        # Class 1 ← red
        w[1], b[1] = [4.0, 0.0, 0.0], -1.0
        # Class 2 ← green
        w[2], b[2] = [0.0, 4.0, 0.0], -1.0
        # Class 3 ← blue
        w[3], b[3] = [0.0, 0.0, 4.0], -1.0
    # head_1: class 1 only when both red and green are high (yellow): 4R + 4G - 5
    if head == 1:
        # Class 1 ← yellow
        w[1], b[1] = [4.0, 4.0, 0.0], -5.0
    # Flattened [C,3,1,1] and [C]
    return [v for row in w for v in row], b


def seg_model(input_name="image", output_names=None, with_scores=True):
    """Segmentation model; the arguments create the broken variants."""
    # Default output names, in the order SegInferenceEngine expects
    names = output_names or ["head_0", "head_1", "head_2", "head_3", "correlation_scores"]
    # NHWC → NCHW
    nodes = [node("Transpose", [input_name], ["x"], [attr_ints("perm", [0, 3, 1, 2])])]
    # Weights of every head
    inits = []
    # Graph outputs
    outputs = []
    # One Conv per head
    for head in range(4):
        # Weights and bias
        w, b = head_weights(head)
        # Weight tensor [C,3,1,1]
        inits.append(tensor("w%d" % head, [HEAD_CLASSES[head], 3, 1, 1], FLOAT, w))
        # Bias tensor [C]
        inits.append(tensor("b%d" % head, [HEAD_CLASSES[head]], FLOAT, b))
        # head logits [1,C,512,512]
        nodes.append(node("Conv", ["x", "w%d" % head, "b%d" % head], [names[head]]))
        # Declare the output
        outputs.append(value_info(names[head], FLOAT, [1, HEAD_CLASSES[head], 512, 512]))
    # correlation_scores
    if with_scores:
        # Constant [1,7] = 0.1 .. 0.7
        scores = tensor("scores", [1, 7], FLOAT, [0.1 * (i + 1) for i in range(7)])
        # Constant node
        nodes.append(node("Constant", [], [names[4]], [attr_tensor("value", scores)]))
        # Declare the output
        outputs.append(value_info(names[4], FLOAT, [1, 7]))
    # Single input
    inputs = [value_info(input_name, FLOAT, [1, 512, 512, 3])]
    # Encode
    return model(nodes, inputs, outputs, inits)


def rsd_pre_model(hist_count_out="hist_count_out"):
    """RSD preprocessing model; a different output name creates the broken variant."""
    # Constants
    inits = [tensor("inv255", [], FLOAT, [1.0 / 255.0]), tensor("one_f", [], FLOAT, [1.0]),
             tensor("one_i", [], INT64, [1])]
    # Graph
    nodes = [
        # uint8 → float
        node("Cast", ["frame"], ["frame_f"], [attr_int("to", FLOAT)]),
        # / 255
        node("Mul", ["frame_f", "inv255"], ["image"]),
        # Colour history + 1 every step
        node("Add", ["hist_stats", "one_f"], ["hist_stats_out"]),
        # Step counter + 1
        node("Add", ["hist_count", "one_i"], [hist_count_out]),
    ]
    # Inputs as in preprocess_step.onnx
    inputs = [value_info("frame", UINT8, [1, 3, 224, 224]), value_info("hist_stats", FLOAT, [10, 1, 3, 4]),
              value_info("hist_count", INT64, [1])]
    # Outputs
    outputs = [value_info("image", FLOAT, [1, 3, 224, 224]), value_info("hist_stats_out", FLOAT, [10, 1, 3, 4]),
               value_info(hist_count_out, INT64, [1])]
    # Encode
    return model(nodes, inputs, outputs, inits)


def rsd_model(phase_output="phase_probs"):
    """RSD model; a different phase output name creates the broken variant."""
    # Fixed phase probabilities: phase 2 (ClippingCutting) wins with 0.7
    probs = [0.05, 0.05, 0.7, 0.05, 0.05, 0.05, 0.05]
    # Constants
    inits = [tensor("probs", [1, 7], FLOAT, probs), tensor("z", [1, 512], FLOAT, [i / 512.0 for i in range(512)]),
             tensor("k60", [], FLOAT, [60.0]), tensor("one_f", [], FLOAT, [1.0])]
    # Graph
    nodes = [
        # Phase probabilities
        node("Identity", ["probs"], [phase_output]),
        # Mean of the carried conv_state (0 on the first step, 1 on the second, ...)
        node("ReduceMean", ["conv_state"], ["state_mean"], [attr_int("keepdims", 0)]),
        # 60 * mean
        node("Mul", ["state_mean", "k60"], ["rsd_part"]),
        # + 60 → 60 s, 120 s, 180 s ...
        node("Add", ["rsd_part", "k60"], ["pred_rsd"]),
        # Mean brightness of the image (0 for a black frame, 1 for a white one)
        node("ReduceMean", ["image"], ["pred_progress"], [attr_int("keepdims", 0)]),
        # Embedding
        node("Identity", ["z"], ["z_t"]),
        # State + 1 every step
        node("Add", ["conv_state", "one_f"], ["conv_state_out"]),
        # Unchanged
        node("Identity", ["ssm_state"], ["ssm_state_out"]),
    ]
    # Inputs as in multimodal_rsd_step.onnx
    inputs = [value_info("image", FLOAT, [1, 3, 224, 224]), value_info("conv_state", FLOAT, [4, 1, 1024, 3]),
              value_info("ssm_state", FLOAT, [4, 1, 16, 64, 16])]
    # Outputs (scalars have unknown rank here; the C++ code reads element 0)
    outputs = [value_info(phase_output, FLOAT, [1, 7]), value_info("pred_rsd", FLOAT),
               value_info("pred_progress", FLOAT), value_info("z_t", FLOAT, [1, 512]),
               value_info("conv_state_out", FLOAT, [4, 1, 1024, 3]),
               value_info("ssm_state_out", FLOAT, [4, 1, 16, 64, 16])]
    # Encode
    return model(nodes, inputs, outputs, inits)


def main():
    """Write every model into the folder given on the command line."""
    # Output folder
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "fake_models"
    # Create it
    os.makedirs(out_dir, exist_ok=True)
    # File name → bytes
    files = {
        # Valid models
        "seg.onnx": seg_model(),
        "rsd_pre.onnx": rsd_pre_model(),
        "rsd.onnx": rsd_model(),
        # Input not called "image"
        "seg_bad_input.onnx": seg_model(input_name="input"),
        # Only 4 outputs (no correlation_scores)
        "seg_missing_scores.onnx": seg_model(with_scores=False),
        # head_0 renamed
        "seg_bad_head_name.onnx": seg_model(output_names=["mask_0", "head_1", "head_2", "head_3",
                                                         "correlation_scores"]),
        # Last output renamed
        "seg_bad_scores_name.onnx": seg_model(output_names=["head_0", "head_1", "head_2", "head_3", "scores"]),
        # hist_count_out missing → Run fails when the C++ code asks for it
        "rsd_pre_bad.onnx": rsd_pre_model(hist_count_out="count_out"),
        # phase_probs missing
        "rsd_bad.onnx": rsd_model(phase_output="probs_out"),
    }
    # Write each
    for name, data in files.items():
        # Binary file
        with open(os.path.join(out_dir, name), "wb") as fh:
            # Model bytes
            fh.write(data)
    # Report
    print("wrote %d fake models to %s" % (len(files), out_dir))


# Run as a script
if __name__ == "__main__":
    # Entry point
    main()
