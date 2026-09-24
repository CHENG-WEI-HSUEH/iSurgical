# syntax=docker/dockerfile:1
# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu AS builder

WORKDIR /workspace/iSurgical

RUN --network=host apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libopencv-dev \
        libyaml-cpp-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgles2 \
        libegl1 \
        libprotobuf-dev \
        protobuf-compiler \
        libgrpc++-dev \
        protobuf-compiler-grpc && \
    rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt ./
COPY src ./src

RUN mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --parallel "$(nproc)"

# Collect all runtime library dependencies and GStreamer plugins
RUN mkdir -p /runtime-libs /runtime-gst-plugins && \
    ldd build/iSurgical | grep "=> /" | awk '{print $3}' | \
    xargs -I{} cp -L {} /runtime-libs/ 2>/dev/null; \
    cp -a /usr/lib/aarch64-linux-gnu/gstreamer-1.0/ /runtime-gst-plugins/; \
    for lib in /usr/lib/aarch64-linux-gnu/gstreamer-1.0/*.so; do \
        ldd "$lib" 2>/dev/null | grep "=> /" | awk '{print $3}' | \
        xargs -I{} cp -Ln {} /runtime-libs/ 2>/dev/null; \
    done; true

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu AS runtime

WORKDIR /workspace/iSurgical

RUN --network=host apt-get update && \
    apt-get install -y --no-install-recommends \
        rsync \
        openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Copy all resolved runtime libraries and GStreamer plugins from builder
COPY --from=builder /runtime-libs/ /usr/lib/aarch64-linux-gnu/
COPY --from=builder /runtime-gst-plugins/gstreamer-1.0/ /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
RUN ldconfig

RUN mkdir -p ./bin ./data/outputs ./data/reports ./logs

# Baked in so the image is a self-contained deliverable (see scripts/package.sh): it runs
# with no bind mounts at all. run.sh still mounts host configs/ and data/ over these when
# they exist, so development keeps working off the working tree.
# Ordered least- to most-frequently-changed to keep the ~5 GB data layers out of rebuilds.
COPY data/models ./data/models
COPY data/inputs ./data/inputs
COPY configs ./configs

COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

COPY --from=builder /workspace/iSurgical/build/iSurgical ./bin/iSurgical

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

ENTRYPOINT ["./entrypoint.sh"]
CMD ["./bin/iSurgical", "configs/app.yaml"]