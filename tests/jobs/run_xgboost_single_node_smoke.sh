#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=tests/jobs/xgboost-single-node-smoke.env
source "$ROOT_DIR/tests/jobs/xgboost-single-node-smoke.env"

RUN_ID="${GITHUB_RUN_ID:-local-$$}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
CONTAINER_NAME="tributo-xgboost-single-node-smoke-${RUN_ID}-${RUN_ATTEMPT}"
IMAGE_TAG="tributo-xgboost-single-node-smoke:${RUN_ID}-${RUN_ATTEMPT}"
LOG_DIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/tributo-xgboost-single-node-smoke-${RUN_ID}-${RUN_ATTEMPT}"
RESULT_DIR=/tmp/tributo-xgboost-single-node-smoke

mkdir -p "$LOG_DIR"

docker_cli() {
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    docker "$@"
}

cleanup() {
  local status=$?
  set +e

  if docker_cli container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker_cli logs "$CONTAINER_NAME" >"$LOG_DIR/container.log" 2>&1
    docker_cli container rm --force "$CONTAINER_NAME" >/dev/null 2>&1
  fi
  if docker_cli image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    docker_cli image rm "$IMAGE_TAG" >"$LOG_DIR/image-cleanup.log" 2>&1
  fi

  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! docker_cli build \
  --build-arg "RAY_IMAGE=$RAY_IMAGE" \
  --tag "$IMAGE_TAG" \
  --file "$ROOT_DIR/tests/jobs/Dockerfile.xgboost-single-node-smoke" \
  "$ROOT_DIR" >"$LOG_DIR/image-build.log" 2>&1; then
  cat "$LOG_DIR/image-build.log"
  exit 1
fi

if ! docker_cli run \
  --detach \
  --name "$CONTAINER_NAME" \
  --label "org.tributo.ci-run=$RUN_ID" \
  --shm-size=512m \
  "$IMAGE_TAG" \
  bash -c 'ray start --head --num-cpus=3 --dashboard-host=127.0.0.1 --disable-usage-stats && sleep infinity' \
  >"$LOG_DIR/container-start.log" 2>&1; then
  cat "$LOG_DIR/container-start.log"
  exit 1
fi

ready=0
for ((attempt = 0; attempt < 30; attempt++)); do
  if docker_cli exec "$CONTAINER_NAME" python -c \
    'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8265/api/version", timeout=2).read()' \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" != 1 ]]; then
  echo "Ray Jobs dashboard did not become ready in the single-node container." >&2
  exit 1
fi

if ! docker_cli exec "$CONTAINER_NAME" ray job submit \
  --address http://127.0.0.1:8265 \
  --working-dir /opt/tributo-smoke \
  -- python xgboost_single_node_smoke.py \
  --root "$RESULT_DIR" \
  --expected-python-version "$PYTHON_VERSION" \
  --expected-tributo-version "$TRIBUTO_CORE_VERSION" \
  --expected-ray-version "$RAY_VERSION" \
  --expected-ray-image "$RAY_IMAGE" \
  >"$LOG_DIR/ray-job.log" 2>&1; then
  cat "$LOG_DIR/ray-job.log"
  exit 1
fi

cat "$LOG_DIR/ray-job.log"
docker_cli cp \
  "$CONTAINER_NAME:$RESULT_DIR/result.json" \
  "$LOG_DIR/result.json"
