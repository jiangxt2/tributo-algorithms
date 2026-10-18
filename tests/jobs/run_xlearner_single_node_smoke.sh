#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=tests/jobs/xlearner-single-node-smoke.env
source "$ROOT_DIR/tests/jobs/xlearner-single-node-smoke.env"

RUN_ID="${GITHUB_RUN_ID:-local-$$}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
RUN_TOKEN="${RUN_ID}-${RUN_ATTEMPT}-$$"
CONTAINER_NAME="tributo-xlearner-single-node-smoke-${RUN_TOKEN}"
IMAGE_TAG="tributo-xlearner-single-node-smoke:${RUN_TOKEN}"
LOG_DIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/tributo-xlearner-single-node-smoke-${RUN_ID}-${RUN_ATTEMPT}"
RESULT_DIR="/tmp/tributo-xlearner-single-node-smoke-${RUN_ID}-${RUN_ATTEMPT}"
IMAGE_BUILD_ATTEMPTED=0
CONTAINER_RUN_ATTEMPTED=0
IMAGE_ID=""

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

  if [[ "$CONTAINER_RUN_ATTEMPTED" == 1 ]] && \
    docker_cli container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker_cli container inspect --format '{{.Id}} {{.Name}} {{.Config.Labels}}' \
      "$CONTAINER_NAME" >"$LOG_DIR/container-identity.log" 2>&1
    container_run_token="$(
      docker_cli container inspect \
        --format '{{ index .Config.Labels "org.tributo.ci-run" }}' \
        "$CONTAINER_NAME" 2>/dev/null
    )"
    if [[ "$container_run_token" == "$RUN_TOKEN" ]]; then
      docker_cli logs "$CONTAINER_NAME" >"$LOG_DIR/container.log" 2>&1
      docker_cli cp "$CONTAINER_NAME:/tmp/ray/session_latest/logs" \
        "$LOG_DIR/ray-session-logs" \
        >"$LOG_DIR/ray-session-logs-copy.log" 2>&1
      docker_cli container rm --force "$CONTAINER_NAME" >/dev/null 2>&1
    else
      echo "Preserving container whose run label does not match this invocation." \
        >"$LOG_DIR/container-cleanup.log"
    fi
  fi
  if [[ "$IMAGE_BUILD_ATTEMPTED" == 1 ]] && \
    docker_cli image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    current_image_id="$(docker_cli image inspect --format '{{.Id}}' "$IMAGE_TAG" 2>/dev/null)"
    image_run_token="$(
      docker_cli image inspect \
        --format '{{ index .Config.Labels "org.tributo.ci-run" }}' \
        "$IMAGE_TAG" 2>/dev/null
    )"
    docker_cli image inspect --format '{{.Id}} {{.RepoTags}}' "$IMAGE_TAG" \
      >"$LOG_DIR/project-image.log" 2>&1
    if [[ "$image_run_token" == "$RUN_TOKEN" ]] && \
      [[ -z "$IMAGE_ID" || "$current_image_id" == "$IMAGE_ID" ]]; then
      docker_cli image rm "$IMAGE_TAG" >"$LOG_DIR/image-cleanup.log" 2>&1
    else
      echo "Preserving image tag whose owner or ID does not match this invocation." \
        >"$LOG_DIR/image-cleanup.log"
    fi
  fi

  docker_cli image ls --filter dangling=true --no-trunc \
    >"$LOG_DIR/dangling-images-after.log" 2>&1
  docker_cli system df >"$LOG_DIR/docker-system-df-after.log" 2>&1
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! docker_cli image ls --filter dangling=true --no-trunc \
  >"$LOG_DIR/dangling-images-before.log" 2>&1; then
  cat "$LOG_DIR/dangling-images-before.log"
  exit 1
fi
if ! docker_cli system df >"$LOG_DIR/docker-system-df-before.log" 2>&1; then
  cat "$LOG_DIR/docker-system-df-before.log"
  exit 1
fi

if docker_cli container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Refusing to reuse existing container name: $CONTAINER_NAME" >&2
  exit 1
fi
if docker_cli image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "Refusing to reuse existing image tag: $IMAGE_TAG" >&2
  exit 1
fi

RAY_BUILD_IMAGE="$RAY_IMAGE"
if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
  RAY_BUILD_IMAGE="$RAY_LOCAL_TAG"
  if ! docker_cli image inspect "$RAY_LOCAL_TAG" >/dev/null 2>&1; then
    echo "The pinned Ray image must be pulled and tagged locally before the manual Gate." >&2
    exit 1
  fi
  RAY_REPO_DIGESTS="$(docker_cli image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$RAY_LOCAL_TAG")"
  EXPECTED_RAY_DIGEST="${RAY_IMAGE##*@sha256:}"
  case "$RAY_REPO_DIGESTS" in
    *"@sha256:$EXPECTED_RAY_DIGEST"*) ;;
    *)
      echo "The local Ray image tag does not contain the pinned manifest digest." >&2
      exit 1
      ;;
  esac
  docker_cli image inspect --format '{{.Id}} {{.RepoTags}} {{.RepoDigests}}' \
    "$RAY_LOCAL_TAG" >"$LOG_DIR/base-image-identity.log" 2>&1
fi

IMAGE_BUILD_ATTEMPTED=1
if ! docker_cli build \
  --label "org.tributo.ci-run=$RUN_TOKEN" \
  --build-arg "RAY_IMAGE=$RAY_BUILD_IMAGE" \
  --tag "$IMAGE_TAG" \
  --file "$ROOT_DIR/tests/jobs/Dockerfile.xlearner-single-node-smoke" \
  "$ROOT_DIR" >"$LOG_DIR/image-build.log" 2>&1; then
  cat "$LOG_DIR/image-build.log"
  exit 1
fi
docker_cli image inspect --format '{{.Id}} {{.RepoTags}}' "$IMAGE_TAG" \
  >"$LOG_DIR/project-image.log" 2>&1
IMAGE_ID="$(docker_cli image inspect --format '{{.Id}}' "$IMAGE_TAG")"

CONTAINER_ARGS=(
  --detach
  --name "$CONTAINER_NAME"
  --label "org.tributo.ci-run=$RUN_TOKEN"
  --shm-size=512m
)
JOB_ARGS=(
  --root "$RESULT_DIR"
  --expected-python-version "$PYTHON_VERSION"
  --expected-tributo-version "$TRIBUTO_CORE_VERSION"
  --expected-ray-version "$RAY_VERSION"
  --expected-ray-image "$RAY_IMAGE"
)

if [[ -n "${XLEARNER_ACIC_DATA_FILE:-}" ]]; then
  : "${XLEARNER_ACIC_DATA_SHA256:?set the SHA-256 of the prepared ACIC input file}"
  : "${XLEARNER_ACIC_SETTING_ID:?set the ACIC 2016 setting ID}"
  : "${XLEARNER_ACIC_REPLICATION:?set the ACIC replication ID}"
  : "${XLEARNER_ACIC_SPLIT_SEED:=7}"
  if [[ "$XLEARNER_ACIC_DATA_FILE" != /* || ! -f "$XLEARNER_ACIC_DATA_FILE" ]]; then
    echo "XLEARNER_ACIC_DATA_FILE must be an existing absolute file path." >&2
    exit 1
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    XLEARNER_ACIC_DATA_FILE="$(cd -- "$(dirname -- "$XLEARNER_ACIC_DATA_FILE")" && pwd)/$(basename -- "$XLEARNER_ACIC_DATA_FILE")"
  fi
  ACIC_BASENAME="$(basename -- "$XLEARNER_ACIC_DATA_FILE")"
  CONTAINER_ARGS+=(--volume "$XLEARNER_ACIC_DATA_FILE:/opt/acic/$ACIC_BASENAME:ro")
  JOB_ARGS+=(
    --acic-data "/opt/acic/$ACIC_BASENAME"
    --acic-sha256 "$XLEARNER_ACIC_DATA_SHA256"
    --acic-setting-id "$XLEARNER_ACIC_SETTING_ID"
    --acic-replication "$XLEARNER_ACIC_REPLICATION"
    --split-seed "$XLEARNER_ACIC_SPLIT_SEED"
  )
elif [[ -n "${XLEARNER_ACIC_DATA_SHA256:-}${XLEARNER_ACIC_SETTING_ID:-}${XLEARNER_ACIC_REPLICATION:-}" ]]; then
  echo "ACIC metadata variables require XLEARNER_ACIC_DATA_FILE." >&2
  exit 1
fi

CONTAINER_RUN_ATTEMPTED=1
if ! docker_cli run "${CONTAINER_ARGS[@]}" "$IMAGE_TAG" \
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
  -- python xlearner_single_node_smoke.py "${JOB_ARGS[@]}" \
  >"$LOG_DIR/ray-job.log" 2>&1; then
  cat "$LOG_DIR/ray-job.log"
  exit 1
fi

cat "$LOG_DIR/ray-job.log"
docker_cli cp "$CONTAINER_NAME:$RESULT_DIR/result.json" "$LOG_DIR/result.json"
