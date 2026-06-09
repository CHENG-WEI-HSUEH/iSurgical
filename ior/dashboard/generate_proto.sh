#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/../../protos"
OUT_DIR="$SCRIPT_DIR/generated"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/surgical_phase.proto"

# Fix imports to use relative within the package
sed -i 's/^import surgical_phase_pb2/from . import surgical_phase_pb2/' "$OUT_DIR/surgical_phase_pb2_grpc.py"

echo "Generated Python gRPC stubs in $OUT_DIR"
