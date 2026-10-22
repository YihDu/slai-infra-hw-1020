#!/bin/sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
docker --context "${SLAI_DOCKER_CONTEXT:-colima-slai-compiler}" build \
  --build-arg HTTP_PROXY="${SLAI_BUILD_PROXY:-http://192.168.5.2:25662}" \
  --build-arg HTTPS_PROXY="${SLAI_BUILD_PROXY:-http://192.168.5.2:25662}" \
  -t slai-gb10-compiler:0.1.15 -f "$task_root/setup/compiler.Dockerfile" "$task_root/setup"
