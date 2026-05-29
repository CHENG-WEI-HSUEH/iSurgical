# Build Project
docker build --no-cache --network=host -f docker/Dockerfile -t isurgical-app:latest .

# Run Project
  docker run --rm -it \
      --runtime nvidia \
      --privileged \
      --network host \
      --ipc=host \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -e DISPLAY=:1 \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      --device /dev/video0:/dev/video0 \
      -v $(pwd)/configs:/workspace/iSurgical/configs \
      isurgical-app:latest \
      ./bin/iSurgical configs/app.yaml
      
      
