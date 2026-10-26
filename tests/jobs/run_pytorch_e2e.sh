#!/usr/bin/env bash
# Run only this invocation's images, services, data and checkpoints.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/tests/jobs/pytorch-e2e.env"
: "${TRIBUTO_CORE_DIR:?Set TRIBUTO_CORE_DIR to the approved Core worktree}"
export TRIBUTO_CORE_DIR
: "${TRIBUTO_E2E_EVIDENCE:?Set TRIBUTO_E2E_EVIDENCE to a new evidence directory}"
[[ ! -e "$TRIBUTO_E2E_EVIDENCE" ]] || { echo 'Evidence directory must be new' >&2; exit 2; }
[[ "$(git -C "$ROOT_DIR" rev-parse --show-toplevel)" == "$ROOT_DIR" ]]
[[ "$(git -C "$TRIBUTO_CORE_DIR" rev-parse --show-toplevel)" == "$TRIBUTO_CORE_DIR" ]]
PROJECT="tributo-pytorch-e2e-$(date -u +%Y%m%d%H%M%S)-$$"
export TRIBUTO_E2E_IMAGE="$PROJECT:local"
export TRIBUTO_E2E_JOB="$ROOT_DIR/tests/jobs/pytorch_e2e_job.py"
export TRIBUTO_E2E_TESTS="$ROOT_DIR/tests"
export TRIBUTO_CORE_VERSION RAY_VERSION TORCH_VERSION MINIO_IMAGE
export TRIBUTO_E2E_ACCESS="$(python3 -c 'import secrets; print(secrets.token_hex(12))')"
export TRIBUTO_E2E_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
mkdir -p "$TRIBUTO_E2E_EVIDENCE/results"
export TRIBUTO_E2E_EVIDENCE
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tributo-pytorch-e2e-build.XXXXXXXX")"
DOCKER=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy docker)
TOUCHED=0
IMAGE_CREATED=0
RAY_ALIAS_CREATED=0
BASE_CAPTURED=0
MONITOR_PID=''
MAIN_FINISHED=0
compose() {
  "${DOCKER[@]}" compose --project-name "$PROJECT" \
    --file "$ROOT_DIR/tests/jobs/docker-compose.pytorch-e2e.yml" "$@"
}
snapshots() {
  "${DOCKER[@]}" image ls --all --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' >"$TRIBUTO_E2E_EVIDENCE/$1-images.txt"
  "${DOCKER[@]}" image ls --filter dangling=true --no-trunc --quiet | sort -u >"$TRIBUTO_E2E_EVIDENCE/$1-dangling.txt"
  "${DOCKER[@]}" system df >"$TRIBUTO_E2E_EVIDENCE/$1-docker-df.txt"
}
cleanup() {
  local status=$? cleanup_status=0 remaining='' image_id=''
  trap - EXIT INT TERM
  set +e
  if [[ "$status" == 0 && "$MAIN_FINISHED" == 0 ]]; then status=1; fi
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
  fi
  if [[ "$TOUCHED" == 1 ]]; then
    compose logs --no-color --timestamps >"$TRIBUTO_E2E_EVIDENCE/services.log" 2>&1
    # Results are a run-owned host bind mount and survive even a failed Job.
    compose down --volumes --remove-orphans --timeout 30 >>"$TRIBUTO_E2E_EVIDENCE/cleanup.log" 2>&1 || cleanup_status=1
  fi
  if [[ "$IMAGE_CREATED" == 1 ]]; then
    image_id="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "$TRIBUTO_E2E_IMAGE" 2>/dev/null)"
    "${DOCKER[@]}" image rm "$TRIBUTO_E2E_IMAGE" >>"$TRIBUTO_E2E_EVIDENCE/cleanup.log" 2>&1 || cleanup_status=1
  fi
  if [[ "$RAY_ALIAS_CREATED" == 1 ]] && "${DOCKER[@]}" image inspect "$RAY_BASE_LOCAL" >/dev/null 2>&1; then
    "${DOCKER[@]}" image rm "$RAY_BASE_LOCAL" >>"$TRIBUTO_E2E_EVIDENCE/cleanup.log" 2>&1 || cleanup_status=1
  fi
  for kind in container network volume; do
    if [[ "$kind" == container ]]; then
      remaining="$("${DOCKER[@]}" container ls --all --quiet --filter "label=com.docker.compose.project=$PROJECT" 2>>"$TRIBUTO_E2E_EVIDENCE/cleanup.log")" || cleanup_status=1
    else
      remaining="$("${DOCKER[@]}" "$kind" ls --quiet --filter "label=com.docker.compose.project=$PROJECT" 2>>"$TRIBUTO_E2E_EVIDENCE/cleanup.log")" || cleanup_status=1
    fi
    [[ -z "$remaining" ]] || cleanup_status=1
    printf '%s=%s\n' "$kind" "$remaining" >>"$TRIBUTO_E2E_EVIDENCE/cleanup.log"
  done
  [[ "$BASE_CAPTURED" == 0 ]] || snapshots final || cleanup_status=1
  "${DOCKER[@]}" ps --all --format '{{.ID}} {{.Names}} {{.State}}' >"$TRIBUTO_E2E_EVIDENCE/final-containers.txt"
  # No rm or broad cache cleanup: this exact directory contains only this run's build inputs.
  trash "$BUILD_DIR" >>"$TRIBUTO_E2E_EVIDENCE/cleanup.log" 2>&1 || cleanup_status=1
  if [[ -d "$TRIBUTO_E2E_EVIDENCE/results/reusable" ]]; then
    python3 - "$TRIBUTO_E2E_EVIDENCE/results/reusable" "$TRIBUTO_E2E_EVIDENCE/temporary-artifact-hashes.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
items = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
Path(sys.argv[2]).write_text(json.dumps(items, indent=2))
PY
    trash "$TRIBUTO_E2E_EVIDENCE/results/reusable" >>"$TRIBUTO_E2E_EVIDENCE/cleanup.log" 2>&1 || cleanup_status=1
  fi
  python3 - "$TRIBUTO_E2E_EVIDENCE/cleanup.json" "$PROJECT" "$status" "$cleanup_status" "$image_id" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(project=sys.argv[2], test_exit=int(sys.argv[3]), cleanup_exit=int(sys.argv[4]), test_image_id=sys.argv[5], temporary_data='no owned resources remain' if sys.argv[4]=='0' else 'check cleanup.log'), indent=2))
PY
  echo "Evidence: $TRIBUTO_E2E_EVIDENCE; test_exit=$status cleanup_exit=$cleanup_status"
  [[ "$status" == 0 ]] || exit "$status"
  exit "$cleanup_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

snapshots baseline
BASE_CAPTURED=1
"${DOCKER[@]}" ps --all --format '{{.ID}} {{.Names}} {{.State}}' >"$TRIBUTO_E2E_EVIDENCE/baseline-containers.txt"
git -C "$ROOT_DIR" rev-parse HEAD >"$TRIBUTO_E2E_EVIDENCE/algorithms-revision.txt"
git -C "$TRIBUTO_CORE_DIR" rev-parse HEAD >"$TRIBUTO_E2E_EVIDENCE/core-revision.txt"
shasum -a 256 "$ROOT_DIR"/tests/jobs/*pytorch* "$ROOT_DIR/tests/test_pytorch_e2e_harness.py" >"$TRIBUTO_E2E_EVIDENCE/harness-sha256.txt"
shasum -a 256 "$TRIBUTO_CORE_DIR/tests/algorithms/test_worker_environment.py" \
  "$TRIBUTO_CORE_DIR/tests/algorithms/test_torch_recipe_worker.py" \
  >"$TRIBUTO_E2E_EVIDENCE/core-test-sha256.txt"
mkdir -p "$BUILD_DIR/wheels"
uv build "$TRIBUTO_CORE_DIR" --project "$TRIBUTO_CORE_DIR" --wheel --no-sources --out-dir "$BUILD_DIR/wheels" >"$TRIBUTO_E2E_EVIDENCE/core-build.log" 2>&1
uv build "$ROOT_DIR" --project "$ROOT_DIR" --wheel --no-sources --out-dir "$BUILD_DIR/wheels" >"$TRIBUTO_E2E_EVIDENCE/algorithm-build.log" 2>&1
python3 "$TRIBUTO_CORE_DIR/tools/host_uv_export.py" --cpu --platform linux/arm64 \
  --extra graph --extra registry --extra grpc --group dev \
  --output-file "$BUILD_DIR/requirements.txt" >"$TRIBUTO_E2E_EVIDENCE/dependency-export.log" 2>&1
if [[ -n "${TRIBUTO_E2E_DEPENDENCY_EXPORT:-}" ]]; then
  python3 - "$BUILD_DIR/requirements.txt" "$TRIBUTO_E2E_DEPENDENCY_EXPORT" <<'PY'
import sys
from pathlib import Path
fresh, cached = map(Path, sys.argv[1:])
def requirements(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
if requirements(fresh) != requirements(cached):
    raise SystemExit('Cached export differs from the fresh locked requirements')
fresh.write_bytes(cached.read_bytes())
PY
  echo 'Verified cached export matches every requirement, marker and hash' >"$TRIBUTO_E2E_EVIDENCE/dependency-cache.log"
fi
cp "$BUILD_DIR/requirements.txt" "$TRIBUTO_E2E_EVIDENCE/requirements.txt"
shasum -a 256 "$BUILD_DIR"/wheels/*.whl >"$TRIBUTO_E2E_EVIDENCE/wheel-sha256.txt"
export TRIBUTO_E2E_REUSE_ENABLED=0
if [[ -n "${TRIBUTO_E2E_REUSE_RESULTS:-}" ]]; then
  python3 - "$TRIBUTO_E2E_EVIDENCE/wheel-sha256.txt" "$TRIBUTO_E2E_REUSE_RESULTS/../wheel-sha256.txt" <<'PY'
import sys
from pathlib import Path
def wheels(path):
    return {Path(name).name: digest for digest, name in (line.split(maxsplit=1) for line in Path(path).read_text().splitlines())}
if wheels(sys.argv[1]) != wheels(sys.argv[2]):
    raise SystemExit('Refusing model reuse across different Core/algorithm Wheels')
fresh = Path(sys.argv[1]).with_name('requirements.txt')
previous = Path(sys.argv[2]).with_name('requirements.txt')
if fresh.read_bytes() != previous.read_bytes():
    raise SystemExit('Refusing model reuse across different locked requirements')
PY
  export TRIBUTO_E2E_REUSE_RESULTS
  TRIBUTO_E2E_REUSE_ENABLED=1
fi
cp "$ROOT_DIR/tests/jobs/Dockerfile.pytorch-e2e" "$BUILD_DIR/Dockerfile"

ensure_image() {
  local image="$1" alias="${2:-}" digest source expected_id actual_id pulled=0
  digest="${image##*@}"
  source="docker.m.daocloud.io/${image%%@*}@$digest"
  if ! "${DOCKER[@]}" image inspect "$image" >/dev/null 2>&1; then
    # Keep a persistent mirror tag, so removing our run alias cannot delete the base.
    "${DOCKER[@]}" pull "${source%%@*}" >>"$TRIBUTO_E2E_EVIDENCE/image-prepare.log" 2>&1
    pulled=1
  else
    source="$image"
  fi
  "${DOCKER[@]}" image inspect --format '{{join .RepoDigests "\n"}}' "$source" | \
    grep -F -- "$digest" >>"$TRIBUTO_E2E_EVIDENCE/image-prepare.log"
  expected_id="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "$source")"
  if [[ "$pulled" == 1 ]]; then
    actual_id="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "${source%%@*}")"
    [[ "$actual_id" == "$expected_id" ]] || { echo 'Mirror tag does not match pinned image' >&2; return 1; }
  fi
  [[ -n "$("${DOCKER[@]}" image inspect --format '{{join .RepoTags " "}}' "$source")" ]] || { echo 'Prepare a persistent pinned base tag before running this gate' >&2; return 1; }
  if [[ -n "$alias" || "$pulled" == 1 ]]; then
    alias="${alias:-${image%%@*}}"
    "${DOCKER[@]}" tag "$source" "$alias"
    actual_id="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "$alias")"
    [[ "$actual_id" == "$expected_id" ]] || { echo 'Pinned image alias ID mismatch' >&2; return 1; }
  fi
  printf '%s %s\n' "$source" "$expected_id" >>"$TRIBUTO_E2E_EVIDENCE/image-prepare.log"
}
ensure_image "$MINIO_IMAGE"
if [[ -n "${TRIBUTO_E2E_BASE_IMAGE:-}" ]]; then
  "${DOCKER[@]}" image inspect "$TRIBUTO_E2E_BASE_IMAGE" >>"$TRIBUTO_E2E_EVIDENCE/image-prepare.log"
  RAY_BASE_LOCAL="$TRIBUTO_E2E_BASE_IMAGE"
else
  RAY_BASE_LOCAL="$PROJECT-ray-base:local"
  ! "${DOCKER[@]}" image inspect "$RAY_BASE_LOCAL" >/dev/null 2>&1 || { echo 'Run-owned base alias already exists' >&2; exit 1; }
  RAY_ALIAS_CREATED=1
  ensure_image "$RAY_IMAGE" "$RAY_BASE_LOCAL"
fi
"${DOCKER[@]}" build --build-arg "RAY_BASE=$RAY_BASE_LOCAL" \
  --build-arg "VENV_PATH=${TRIBUTO_E2E_VENV_PATH:-/opt/tributo-e2e/venv}" \
  --build-arg "PIP_INDEX_URL=${TRIBUTO_E2E_PIP_INDEX_URL:-https://pypi.org/simple}" \
  --build-arg "PIP_DEFAULT_TIMEOUT=${TRIBUTO_E2E_PIP_TIMEOUT:-15}" \
  --label "com.docker.compose.project=$PROJECT" --tag "$TRIBUTO_E2E_IMAGE" \
  "$BUILD_DIR" >"$TRIBUTO_E2E_EVIDENCE/image-build.log" 2>&1
IMAGE_CREATED=1
"${DOCKER[@]}" image inspect --format '{{.Id}}' "$TRIBUTO_E2E_IMAGE" >"$TRIBUTO_E2E_EVIDENCE/runtime-image-id.txt"
TOUCHED=1
compose up --detach --no-build --pull never >"$TRIBUTO_E2E_EVIDENCE/startup.log" 2>&1
compose exec -T --user root ray-head chown -R ray:users /e2e
ready=0
for ((attempt=0; attempt<90; attempt++)); do
  if compose exec -T ray-head python /opt/tributo-e2e/pytorch_e2e_job.py --ready >"$TRIBUTO_E2E_EVIDENCE/readiness.log" 2>&1; then ready=1; break; fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo 'Cluster/services not ready' >&2; exit 1; }
monitor() {
  while true; do
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    local containers=()
    while IFS= read -r container; do containers+=("$container"); done < <(compose ps --quiet)
    if [[ "${#containers[@]}" -gt 0 ]]; then
      "${DOCKER[@]}" stats --no-stream --format '{{json .}}' "${containers[@]}" || true
    fi
    sleep 5
  done
}
monitor >"$TRIBUTO_E2E_EVIDENCE/docker-stats.jsonl" 2>&1 &
MONITOR_PID=$!
compose exec -T ray-head python /opt/tributo-e2e/pytorch_e2e_job.py --orchestrate "$@" \
  2>&1 | tee "$TRIBUTO_E2E_EVIDENCE/orchestrator.log"
MAIN_FINISHED=1
