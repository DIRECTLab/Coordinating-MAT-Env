sudo docker run --name grutopia -it --rm --gpus all --network host \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "WEBUI_HOST=${WEBUI_HOST}" \
  -v ${PWD}:/isaac-sim/GRUtopia \
  -v ${CACHE_ROOT}/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ${CACHE_ROOT}/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ${CACHE_ROOT}/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ${CACHE_ROOT}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ${CACHE_ROOT}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ${CACHE_ROOT}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ${CACHE_ROOT}/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ${CACHE_ROOT}/isaac-sim/documents:/root/Documents:rw \
  grutopia:0.0.1
