# Tributo Official Algorithms

This repository publishes one Python distribution, `tributo-algorithms`. The
Wheel contains all official algorithm descriptors and plugins; optional extras
install the runtime dependencies for each algorithm domain. Third-party
algorithms use the same public contracts and entry-point path.

Tributo Core owns Ray execution, resource allocation, retries, input leases,
checkpoint transport, evidence, Tune, Bundle publication, and inference. This
distribution owns algorithm mathematics, framework-native hooks, executable
contracts, and algorithm-specific delivery plugins.

| Extra | Algorithms and roles |
| --- | --- |
| `classical` | Random Forest, Extra Trees, Logistic/Linear Regression, MultinomialNB, PCA, KMeans, synchronous SGD, Isolation Forest |
| `boosting` | Ray Train XGBoost and LightGBM, ONNX/UBJ export, native flavor |
| `tabular-torch` | DNN, nnPU/uPU, PU prior and metric utilities |
| `timeseries` | Temporal convolution, LSTM, and GRU classification |
| `catboost` | Conditional distributed CatBoost ensemble |
| `representation` | Distributed tabular autoencoder |
| `transformers-nlp` | Pre-tokenized Transformer classification |
| `graph-pyg` | GraphSAGE and relational R-GCN |
| `recsys-torch` | Two-Tower and Jagged EmbeddingBag with All-to-All routing |
| `multistage-torch` | Distillation and pretrain-to-finetune |
| `causal-core` | Difference-in-means, DML, and IV |
| `causal-discovery` | Distributed PC stability discovery |
| `causal-dr` | Doubly robust/AIPW estimation |
| `causal-xlearner` | Five-stage X-Learner and batch CATE flavor |
| `causal-dowhy` | DoWhy estimation/refutation and GCM root-cause analysis |

Install only the domain dependencies you need:

```bash
pip install "tributo-algorithms[classical]"
pip install "tributo-algorithms[causal-xlearner]"
```

Use `tributo-algorithms[all]` to install every domain extra. The base
distribution registers all descriptors, so discovery remains available
without importing the optional algorithm frameworks. Selecting an algorithm
requires installing its corresponding extra.

The `graph-pyg` extra currently supports Torch `>=2.5,<2.14`; `all` follows
the same range. PyG's module import uses TorchScript APIs deprecated in Torch
2.14. Other Torch extras retain their own dependency ranges.

## Core compatibility

The distribution and domain extras require Tributo Core `>=1.2` without an
upper version bound. The XGBoost descriptor also requires `>=1.2` for optional
input routing; other descriptors retain their own minimum versions. Versions
below 1.2 are outside this distribution's declared compatibility range.
An open upper bound does not certify future Core
releases; execution support is established by the Gate for its exact Core
version, algorithm Wheel, and runtime image. Lock those versions in consuming
projects for reproducible deployments.

## macOS arm64 with Python 3.12

DoWhy 0.14 requires SciPy `<=1.15.3` on Python 3.12. The SciPy 1.15.3 macOS 14
Wheel fails dynamic loader validation on affected macOS versions. The official
macOS 12 Wheel of the same version works on these systems. The project's uv
source configuration selects this Wheel for macOS arm64 with Python 3.12 and
records its hash in `uv.lock`.

For pip installations of `causal-dowhy` or `all` on this platform, install that
official SciPy Wheel before the algorithm extra:

```bash
python -m pip install "https://files.pythonhosted.org/packages/7b/7e/f30be3d03de07f25dc0ec926d1681fed5c732d759ac8f51079708c79e680/scipy-1.15.3-cp312-cp312-macosx_12_0_arm64.whl#sha256=185cd3d6d05ca4b44a8f1595af87f9c372bb6acf9c808e99aa3e9aa03bd98cf6"
python -m pip install "tributo-algorithms[causal-dowhy]"
```

## Development

```bash
uv sync --locked --extra all --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy packages
uv run pytest tests
uv build --no-sources
```

Release the aggregate distribution with a single
`tributo-algorithms-v<semver>` tag. CI builds one Wheel and source distribution,
publishes the same checksummed files to TestPyPI, verifies the installed
metadata, then publishes those files to PyPI.

## XGBoost validation and early stopping

XGBoost accepts a required `train` binding and an optional `validation` binding
through `InputBindingSet`, with `train` as the primary role. Both inputs must
declare the same ordered features and label column and provide compatible
numeric schemas. Validation rows are evaluated separately and never enter the
training gradient matrix. Both roles use lossless partitioning and report
their own `input_rows` counts for Core validation.

For validation-driven model selection, include these fields in the existing
`algorithm_config` (alongside its `data`, `ray`, and `output` configuration):

```json
{
  "model": {"objective": "binary:logistic", "eval_metric": "logloss"},
  "training": {
    "num_rounds": 1000,
    "early_stopping_rounds": 20,
    "maximize": false
  }
}
```

Early stopping requires validation, one explicit `model.eval_metric` string,
positive integer patience, and an explicit boolean `training.maximize`.
Patience may exceed the round limit. The native XGBoost callback selects the
best prefix even when the round limit is reached before patience expires.
Linear Boosters cannot export this selected prefix. A provided validation
shard must be non-empty on every Worker, matching the current train-shard
restriction. Invalid inputs or non-finite selection scores fail training.

Omit early-stopping fields to preserve fixed-round training. Validation can
still be provided for evaluation; train-only requests remain supported.
Execution metrics distinguish `training_rounds` (executed rounds) from
`model_rounds` (rounds in the exported model). Early-stopped fits additionally
return zero-based `best_iteration`, `best_score`, `selection_metric`, and
`maximize`. The `train-*` and `validation-*` metrics describe the last executed
round; `best_score` describes the selected round. Both ONNX and native UBJ
artifacts contain the selected model and a digest-checked
`training_metadata.json` sidecar. Best-model selection does
not depend on a consumer supplying an iteration range at prediction time.

For H&M, use validation logloss for stopping. XGBoost's distributed AUC is an
[approximation](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters).
Compute AP, ROC-AUC, and Lift over the complete held-out predictions and labels
in the calling application.

## XGBoost periodic checkpoints and same-run recovery

Enable periodic saving and a finite Worker failure allowance in the existing
configuration:

```json
{
  "training": {"num_rounds": 20, "checkpoint_interval_rounds": 10},
  "ray": {"storage_path": "/shared/checkpoints", "max_failures": 1}
}
```

Both fields default to zero. Values must be non-negative integers; booleans
and infinite retry budgets are rejected. A positive `max_failures` requires
a positive checkpoint interval. Saving without retries is allowed. Ray Train
V2 owns checkpoint transport and Worker group restarts; every trainer invocation
has a unique storage name. All nodes must access the same durable storage path.
Recovery requires `gbtree`, fixed Worker count/environment, and frozen inputs.
The exercised baseline is Ray 2.59.0 with XGBoost 3.4.1 and Core 1.2.0.

`num_rounds` remains the total target: restoring round 10 with a target of 20
adds at most 10 rounds. Checkpoints retain the full training Booster, ordered
features, invocation/plan identity, environment versions, file digests, and
early-stopping score history. Native early stopping replays that history to
retain best score and patience progress. A checkpoint that has reached the
round limit or stopped early adds no further rounds. Corrupt or incompatible
state fails explicitly, without silently restarting from zero. A failure before
the first checkpoint may restart from zero within the configured allowance.

The latest recovery state and selected inference model are separate. Only a
final checkpoint can be exported; both Bundle formats contain the selected
model. Metrics include `resumed_from_round` and `attempt_training_rounds`, whose
sum equals `training_rounds`. Final Worker receipts contain only the successful
group's input counts and newly executed rounds, including zero for terminal
recovery. Failed attempts do not contribute to coverage evidence.

This supports Worker process failures within one `fit()`. Cross-run
`runtime.resume_from`, Driver/Controller or cluster disaster recovery, elastic
Worker counts, and Tune are outside this feature. The public stage-runner API
is unchanged and keeps recovery disabled. Input identity checks do not detect
changed contents behind the same mutable URI. UBJ continuation does not promise
bitwise equivalence for random sampling, GPU training, or changed data shards;
the Gate uses deterministic CPU histogram training without sampling.

## XGBoost Worker Gate

XGBoost consumes replayable Ray Train shards in batches of at most 65,536 rows.
The existing Core Framework Native and multi-stage entry points share this
Worker loop; Core continues to manage execution and Bundle delivery.

PR CI also runs a small single-node Ray Jobs smoke for the installed aggregate
Wheel. Its pinned Python/Core/Ray profile is recorded in
`tests/jobs/xgboost-single-node-smoke.env`. The smoke checks the
end-to-end training and Bundle inference path; it does not validate cross-node
execution. The manual Gate below remains the cross-node validation harness.

The Gate targets a two-node Docker Ray 2.59.0 cluster with the same fixed
Tributo Core 1.2.0 and algorithm Wheels installed on every node, together with the
`boosting` extra. Use a shared directory mounted at the same absolute path on
both nodes. Submit from this repository through Ray Jobs:

```bash
ray job submit --address "$RAY_DASHBOARD_URL" --working-dir . \
  -- python tests/jobs/xgboost_worker_gate.py --root "$GATE_SHARED_ROOT"
```

`GATE_SHARED_ROOT` must name a new directory. The job checks the Core execution
path and `XGBoostStageRunner` with two Workers on different nodes, 131,075 input
train rows, 131,077 validation rows, synchronized model digests, train-only
and validation fits, early stopping, selected-model reload, and ONNX/native
Bundle artifacts. A separate bounded routing fixture verifies row identities
for both input roles, and an empty validation case must fail without publishing
a Bundle. Recovery cases inject real Worker exits after the controller confirms
checkpoint persistence: 10-to-20-round continuation, interruption during
patience, further improvement up to the round cap, final-checkpoint recovery
with no new rounds, finite retry exhaustion, and invocation/checkpoint isolation.
Final receipts are matched to the last successful group. It preserves
`result.json`, failure events, receipts, and model outputs
under that directory. This targeted Gate does not certify other runtime images
or execution profiles.

## X-Learner held-out effect evaluation

PR CI runs the installed aggregate Wheel through a single-node Ray Jobs smoke
using a deterministic truth-known fixture. The smoke exercises X-Learner fit,
loads its Bundle through the batch prediction capability, and reports
unnormalized root PEHE and absolute population ATE error on a held-out cohort.

The manual ACIC 2016 Gate uses a prepared CSV or Parquet file that is not
committed to the repository. It must contain unique `sample_id`, binary
`treatment`, factual `outcome`, expected potential-outcome means `mu0` and
`mu1`, and at least three numeric feature columns named contiguously `x0`,
`x1`, ... . The file must not contain extra columns: unmapped fields are
rejected so potential-outcome truth cannot accidentally become a feature.
Categorical covariates must be encoded numerically before preparation. The
result's `schema_mapping` records the canonical columns in the prepared file;
it does not infer the original ACIC column names or categorical encoding. Keep
those preparation details with the local dataset provenance. The Gate verifies
the checksum of the exact prepared file, separates training rows from held-out
rows with the recorded seed, and excludes `mu0` and `mu1` from training input.

For a manual run, use one fixed ACIC setting and replication, then record the
source, setting, replication, schema mapping, split seed, and prepared-file
SHA-256 in the result. The reported metrics are project-local unnormalized
root PEHE in outcome units and population ATE error over the held-out cohort;
they do not reproduce the ACIC competition's normalized PEHE or SATT scoring.
The dataset source is the [ACIC 2016 competition page](https://jenniferhill7.wixsite.com/acic-2016/competition).

With the pinned Docker/Ray profile available, run the manual Gate as follows:

```bash
uv lock --check
uv build --no-sources
uv export \
  --locked \
  --extra causal-xlearner \
  --no-dev \
  --no-emit-project \
  --no-annotate \
  --output-file dist/xlearner-smoke-requirements.txt

XLEARNER_ACIC_DATA_FILE=/path/to/acic-setting-replication.parquet \
XLEARNER_ACIC_DATA_SHA256="replace-with-prepared-file-sha256" \
XLEARNER_ACIC_SETTING_ID=1 \
XLEARNER_ACIC_REPLICATION=1 \
XLEARNER_ACIC_SPLIT_SEED=7 \
bash tests/jobs/run_xlearner_single_node_smoke.sh
```

The Gate uses Tributo Core 1.2.0, Ray 2.59.0, Python 3.12, and the
digest-pinned Ray image in `tests/jobs/xlearner-single-node-smoke.env`. It
records the runtime, dataset checksum, Bundle identity, metrics, and logs, then
removes only its own container and custom image. The result validates a
single-node path and does not claim online-serving or multi-node support.
For a local manual run, pull the pinned base image through the configured
registry mirror, tag it with the value of `RAY_LOCAL_TAG` in the profile, and
ensure its repository digest matches the profile. The runner verifies the
digest and does not pull the base image from the default registry.
