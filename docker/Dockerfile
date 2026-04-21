# =========================
# Stage 1: Builder
# =========================
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu AS builder

WORKDIR /workspace/iSurgical

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopencv-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests

RUN mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j"$(nproc)"

# =========================
# Stage 2: Runtime
# =========================
FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.3.0-igpu AS runtime

WORKDIR /workspace/iSurgical

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopencv-dev \
        libgles2 \
        libopengl0 \
        libegl1 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ./bin ./data/inputs ./data/outputs ./data/reports ./logs

COPY --from=builder /workspace/iSurgical/build/iSurgical ./bin/iSurgical
COPY configs ./configs
COPY data/models ./data/models

CMD ["./bin/iSurgical"]