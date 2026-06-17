# qcap_fullscreen Troubleshooting Guide

## Overview

This application uses OpenCV + GStreamer to capture video from the Yuan SC0710 capture card
(`/dev/video0`) at 4K@60fps BGR format (~24.8 MB per frame).

---

## Error Messages & Solutions

### Error 1: Failed to allocate required memory

**Full error output:**
```
[ WARN:0@0.074] global cap_gstreamer.cpp:2784 handleMessage OpenCV | GStreamer warning:
  Embedded video playback halted; module v4l2src0 reported: Failed to allocate required memory.
[ WARN:0@0.076] global cap_gstreamer.cpp:1679 open OpenCV | GStreamer warning: unable to start pipeline
[ WARN:0@0.076] global cap_gstreamer.cpp:1164 isPipelinePlaying OpenCV | GStreamer warning:
  GStreamer: pipeline have not been created
[ WARN:0@0.076] global cap.cpp:204 open VIDEOIO(GSTREAMER):
  backend is generally available but can't be used to capture by name
Failed to open QCAP pipeline.
Pipeline:
v4l2src device=/dev/video0 io-mode=mmap do-timestamp=true ! queue max-size-buffers=2 ...
```

**Cause:** CMA (Contiguous Memory Allocator) is exhausted. V4L2 mmap buffers require
contiguous physical memory. At 4K BGR, minimum 3 buffers = ~75 MB CMA needed.

**Diagnosis:**
```bash
# Check CMA availability (need at least 75 MB free)
grep -i cma /proc/meminfo
# Example BAD output:
#   CmaTotal:   262144 kB
#   CmaFree:      6744 kB    <-- not enough!

# Find what is consuming CMA (likely another V4L2 user)
fuser /dev/video0
ps aux | grep -E "iSurgical|gst-launch|qcap"
```

**Solution:**
```bash
# Kill the process holding /dev/video0
sudo kill <PID>

# Verify CMA is freed
grep CmaFree /proc/meminfo
# Should show > 75000 kB

# Retry
./qcap_fullscreen
```

---

### Error 2: Device busy / cannot open device

**Full error output:**
```
[ WARN:0@0.005] global cap_gstreamer.cpp:2784 handleMessage OpenCV | GStreamer warning:
  Embedded video playback halted; module v4l2src0 reported: Cannot identify device '/dev/video0'.
```
or
```
[ WARN:0@0.003] global cap_gstreamer.cpp:2784 handleMessage OpenCV | GStreamer warning:
  Embedded video playback halted; module v4l2src0 reported: Device '/dev/video0' is busy.
Failed to open QCAP pipeline.
```

**Cause:** Another process has exclusively opened `/dev/video0`.

**Diagnosis:**
```bash
# Identify the process
fuser -v /dev/video0
# Example output:
#   /dev/video0:  USER   PID ACCESS COMMAND
#                 root  505746 F.... iSurgical

# Or check more broadly
lsof /dev/video0
```

**Solution:**
```bash
# Stop the other process
sudo kill <PID>

# If the process won't die
sudo kill -9 <PID>

# Verify device is free
fuser /dev/video0
# Should print nothing (exit code 1 = no users)

# Retry
./qcap_fullscreen
```

---

### Error 3: No signal / empty frames after opening

**Full error output:**
```
Frame size: 3840 x 2160
Failed to read frame.
```
(Application opens successfully but immediately fails to read)

**Cause:** The capture card has no input signal (HDMI cable unplugged, source device off,
or wrong input selected).

**Diagnosis:**
```bash
# Check input status
v4l2-ctl -d /dev/video0 --all | grep "Video input"
# Good: Video input : 0 (HDMI k0/TVI/GMSL/SDVOE INPUT(0): ok)
# Bad:  Video input : 0 (HDMI k0/TVI/GMSL/SDVOE INPUT(0): no signal)
```

**Solution:**
1. Check HDMI cable is connected to the source and capture card
2. Ensure the source device is powered on and outputting video
3. Verify correct input is selected:
   ```bash
   v4l2-ctl -d /dev/video0 --get-input
   v4l2-ctl -d /dev/video0 --set-input=0
   ```

---

### Error 4: Permission denied

**Full error output:**
```
[ WARN:0@0.001] global cap_gstreamer.cpp:2784 handleMessage OpenCV | GStreamer warning:
  Embedded video playback halted; module v4l2src0 reported: Cannot open device '/dev/video0',
  Permission denied.
Failed to open QCAP pipeline.
```

**Cause:** Current user is not in the `video` group.

**Diagnosis:**
```bash
# Check device permissions
ls -la /dev/video0
# crw-rw----+ 1 root video 81, 0 ...

# Check if user is in video group
groups
# Should include 'video'
```

**Solution:**
```bash
# Add user to video group
sudo usermod -aG video $USER

# Log out and back in, or use newgrp for current session
newgrp video

# Retry
./qcap_fullscreen
```

---

### Error 5: GStreamer backend not available

**Full error output:**
```
[ WARN:0@0.001] global cap.cpp:204 open VIDEOIO(GSTREAMER): backend is not available
  (plugin/backend is missing or can't be loaded)
Failed to open QCAP pipeline.
```

**Cause:** OpenCV was built without GStreamer support, or GStreamer libraries are missing.

**Diagnosis:**
```bash
# Check if OpenCV has GStreamer support
python3 -c "import cv2; print(cv2.getBuildInformation())" 2>/dev/null | grep -i gstreamer
# Or check shared library dependencies
ldd ./qcap_fullscreen | grep gst

# Verify GStreamer is installed
gst-inspect-1.0 --version
gst-inspect-1.0 v4l2src
```

**Solution:**
```bash
# Install GStreamer runtime dependencies
sudo apt install libgstreamer1.0-0 gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# If OpenCV itself lacks GStreamer, rebuild with -DWITH_GSTREAMER=ON
```

---

### Error 6: Display / window creation failure

**Full error output:**
```
(qcap_fullscreen:12345): Gtk-WARNING **: cannot open display:
```
or
```
: cannot connect to X server
```

**Cause:** No display server available (e.g. running over SSH without X forwarding,
or in a Docker container without display access).

**Diagnosis:**
```bash
echo $DISPLAY
# Should show something like :0 or :1
# Empty = no display configured
```

**Solution:**
```bash
# If running via SSH, enable X forwarding
ssh -X user@host

# If running locally but DISPLAY is unset
export DISPLAY=:0

# If running in Docker, pass display through
docker run --env DISPLAY=$DISPLAY --volume /tmp/.X11-unix:/tmp/.X11-unix ...
```

---

## Quick Diagnostic Script

Run this to check all common failure conditions at once:

```bash
#!/bin/bash
echo "=== Device exists? ==="
ls -la /dev/video0 2>&1

echo ""
echo "=== Device in use? ==="
fuser -v /dev/video0 2>&1 || echo "Device is free"

echo ""
echo "=== CMA Memory ==="
grep -i cma /proc/meminfo

echo ""
echo "=== User groups ==="
groups

echo ""
echo "=== Signal status ==="
v4l2-ctl -d /dev/video0 --all 2>&1 | grep -E "Video input|Width|Pixel Format|Size Image"

echo ""
echo "=== Display ==="
echo "DISPLAY=$DISPLAY"

echo ""
echo "=== Competing processes ==="
ps aux | grep -E "iSurgical|gst-launch|qcap" | grep -v grep
```

---

## Hardware Reference

| Item | Value |
|------|-------|
| Capture card driver | LXV4L2D_SC0710 (Yuan SC0710) |
| Device node | `/dev/video0` |
| Resolution | 3840 x 2160 |
| Frame rate | 60 fps |
| Pixel format | BGR3 (24-bit, 3 bytes/pixel) |
| Frame size | 3840 x 2160 x 3 = 24,883,200 bytes (~24.8 MB) |
| CMA total (system) | 256 MB |
| Min CMA needed | ~75 MB (3 buffers) |
| Platform | Jetson (aarch64, Linux 5.15-tegra) |

## Key Principle

V4L2 mmap capture is **exclusive** — only one process can stream from `/dev/video0` at a time.
The most common failure is that the main `iSurgical` application is already running and holding
all the CMA buffers. Always check `fuser /dev/video0` first.
