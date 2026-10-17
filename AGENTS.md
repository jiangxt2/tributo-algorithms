# Tributo Algorithms — Codex Project Guide

This file contains project-specific facts and working rules for the
`tributo-algorithms` repository. Workspace-wide approval, safety, worktree,
Git, testing, and publishing rules come from `~/GitHub/AGENTS.md`.
Both files are active instructions.

## Mandatory instruction loading

Before any task in this repository or one of its worktrees, fully read:

- `~/GitHub/AGENTS.md`;
- this `AGENTS.md`.

If the task also changes or reviews Tributo Core, fully read
`~/GitHub/Tributo/AGENTS.md` as well. Do not infer the
current contract from a previous session, a plan document, or memory.

## Project role and current baseline

`tributo-algorithms` is the independent first-party algorithm monorepo for
`jiangxt2/tributo-algorithms`. It publishes one Python distribution containing
all domain source packages. Optional extras install the runtime dependencies
for selected algorithm domains. The repository is not Tributo Core and must
not grow a second Core runtime, input gateway, algorithm registry, or CLI
algorithm-name dispatch table.

Tributo Core owns the framework-neutral execution control plane:

- algorithm planning, Worker/Task/Actor lifecycle, resource placement,
  cancellation, retries, and Ray Data sharding;
- execution receipts, input-coverage evidence, state coordination, and
  checkpoint transport;
- Tune trial orchestration and fit-only semantics;
- Bundle planning, publication, reading, model loading, and Ray Data/Ray
  Serve inference.

This repository owns algorithm-domain behavior:

- model mathematics, losses, metrics, estimators, and bounded stage logic;
- framework-native adapters where the selected descriptor explicitly delegates
  distributed control to a public framework API;
- algorithm-specific configuration, input/output/coverage validators,
  exporters, flavors, and model metadata.

The current source baseline contains 15 domain source packages, one root
distribution, 37 `tributo.algorithms` entry points, 9 exporters, 1 validator,
and 3 model flavors. The aggregate distribution currently declares version
`0.1.0`; the root lock resolves Tributo `1.1.0` and Ray `2.55.1`. These facts
describe the checked-in baseline, not a permanent support guarantee. An
algorithm being present, buildable, or discoverable does not by itself mean
that it is `tested`, `validated`, or `supported` in Tributo's support model.

The authoritative current capability sources are the checked-in source,
package metadata, lock file, tests, and the Tributo support/conformance
implementation. Plans under `~/GitHub/plans/` are design or progress context;
do not copy a planned capability into this file as an implemented fact without
matching source, configuration, and test evidence.

## Package and algorithm map

The following map reflects the current package metadata and descriptors. Keep
it synchronized when an algorithm or package is added, removed, renamed, or
changes its primary execution strategy.

| Package | Current algorithms / entry points | Primary Tributo strategy |
| --- | --- | --- |
| `classical` | Random Forest and Extra Trees (`joblib` and native ensemble), Logistic/Linear Regression, MultinomialNB, PCA, KMeans, synchronous SGD, Isolation Forest | `RAY_JOBLIB_ESTIMATOR`, `RAY_PARALLEL_ENSEMBLE`, `RAY_ITERATIVE_OPTIMIZATION`, `RAY_MAP_REDUCE` |
| `boosting` | XGBoost and LightGBM, plus ONNX/UBJ exporters and native validator/flavor | `FRAMEWORK_NATIVE` |
| `catboost` | Conditional distributed CatBoost ensemble | `RAY_PARALLEL_ENSEMBLE` |
| `tabular-torch` | DNN (`dnn`) and positive-unlabeled DNN (`pu`), including PU priors and metrics | `RAY_TRAIN_TORCH` / `TorchRecipe` |
| `timeseries` | Temporal convolution, LSTM, and GRU classifiers | `RAY_TRAIN_TORCH` / `TorchRecipe` |
| `representation` | Tabular autoencoder | `RAY_TRAIN_TORCH` / `TorchRecipe` |
| `transformers-nlp` | Pre-tokenized Transformer classifier | `RAY_TRAIN_TORCH` / `TorchRecipe` |
| `graph-pyg` | GraphSAGE and relational R-GCN node classifiers | `RAY_TRAIN_TORCH` / `RayTorchAdapter` |
| `recsys-torch` | Two-Tower Recipe and jagged-history Adapter with explicit All-to-All routing | `RAY_TRAIN_TORCH` |
| `multistage-torch` | Teacher-to-student distillation and pretrain-then-finetune classifier | `RAY_TRAIN_TORCH` / component Adapter stages |
| `causal-core` | Difference-in-means ATE, linear DML ATE, linear IV ATE | `RAY_MAP_REDUCE` |
| `causal-discovery` | PC stability causal discovery | `RAY_MAP_REDUCE` |
| `causal-dr` | Doubly robust/AIPW ATE with nuisance stages | `FRAMEWORK_NATIVE` with component stages |
| `causal-xlearner` | Five-stage X-Learner and its Bundle exporters/flavor | `FRAMEWORK_NATIVE` with component stages |
| `causal-dowhy` | DoWhy estimate/refute and GCM root-cause attribution | `FRAMEWORK_NATIVE` with component stages |

The package directories are source domains, not separately buildable or
publishable distributions. Every descriptor and artifact producer uses the
`tributo-algorithms` distribution identity. Install the matching optional
extra, such as `tributo-algorithms[causal-core]`, for its runtime dependency
closure.

## Tributo integration boundary

Use Tributo's public algorithm surface:

- `tributo.algorithms` and `tributo.algorithms.api` for builders, descriptors,
  registrations, distribution policies, input bindings, contracts, results,
  and execution metadata;
- `tributo.algorithms.spi` for algorithm hooks such as
  `MapReduceAlgorithm`, `ParallelEnsembleAlgorithm`,
  `IterativeOptimizationAlgorithm`, `FrameworkNativeAlgorithm`, `TorchRecipe`,
  and `RayTorchAdapter`;
- the documented public exporting and flavor protocols for Bundle delivery.

Do not introduce imports of `tributo.algorithms.core`, private Tributo modules,
or private Ray modules. Public Ray APIs may be used by a framework-native
adapter when that is part of the algorithm's declared strategy, but the
adapter must not call `ray.init()` or replace Core's generic retry policy,
input coverage accounting, checkpoint ownership, or execution evidence
contract. A framework-native implementation may configure a public Ray
trainer and create narrowly scoped framework-stage/evidence helpers required
by its descriptor.

Select the SPI that matches the mathematics:

- `MapReduceAlgorithm`: each shard emits a bounded state with a declared
  schema; merge is associative (and commutative where declared), the identity
  handles empty shards, and retry safety is explicit.
- `ParallelEnsembleAlgorithm`: Core schedules deterministic independent units;
  the algorithm plans unit IDs/seeds, fits one unit, merges ordered units, and
  finalizes the ensemble. Unit parallelism is not the same as training one
  estimator on sharded data.
- `IterativeOptimizationAlgorithm`: each round computes shard-local updates
  against one global state, reduces them, applies a global update, evaluates,
  and decides whether to stop. Do not substitute trial parallelism, OvR
  parallelism, or independent local models for this contract.
- `TorchRecipe`: implement typed model/batch/loss/optimization/metric/artifact
  hooks; Core owns the Ray Train loop, sharding, reductions, reporting,
  Stage replay, checkpoint transport, and Bundle publication.
- `RayTorchAdapter`: implement only the framework-specific Worker loop and
  model/artifact hooks. Core still constructs the sole `TorchTrainer`, routes
  role datasets, transports invocation-local Stage dependencies, and publishes
  the Bundle. Runtime API v1 does not support cross-Run recovery.
- `FrameworkNativeAlgorithm`: validate the framework before reading data,
  bind canonical datasets under the Core input contract, avoid unbounded driver
  materialization, build the declared framework trainer, collect actual
  Worker/Node/Rank/Shard/state evidence, and return the framework checkpoint
  consumed by Bundle export. A bounded, explicitly tested materialization is a
  separate algorithm limitation, not a general input pattern.

The current formal Builder path is fit-oriented and uses
`DistributionSpec` plus `ExecutionRequest`. New work must not create a legacy
runtime branch or a second execution protocol. If an algorithm needs a
different lifecycle or operation, first verify that an existing public SPI
expresses it; a Core change requires a separately approved scope.

## Descriptor, entry point, and contract rules

Every algorithm is discovered through the standard Python entry-point group
`tributo.algorithms`. An entry point must resolve to a lightweight descriptor,
not directly to the implementation class. Descriptor import must not start
Ray, access the network, load model weights, or import heavy implementation
frameworks such as Torch, PyG, Transformers, XGBoost, DoWhy, or causal-learn.
Runtime implementation imports should remain lazy where the package exposes a
package-level convenience import.

Current official descriptors use the v2 distributed descriptor and bind four
executable contracts:

- configuration;
- input;
- output;
- distributed coverage.

Each binding must have a versioned contract ID, schema digest, and qualified
validator reference. The descriptor must keep its algorithm identity,
implementation ID, package/distribution name and version, Tributo version
range, execution mode, `DistributionSpec`, worker range, execution profiles,
resource requirements, exporter/flavor identity, and limitations internally
consistent. Contract validators must fail closed on unknown keys, malformed
roles, missing required output, incomplete coverage, invalid stage evidence,
or unsafe input.

The current package baseline uses repeated-character values such as
`"1" * 64` as schema-digest placeholders. Tributo's current binding check
validates the binding/validator match and digest shape; it does not make these
values independent canonical Contract artifacts. Do not present them as
cryptographic schema evidence or as sufficient support evidence. A generated
Contract artifact/digest migration requires its own compatibility and release
review.

Use `AlgorithmBuilder` helpers when they match the strategy:
`from_distributed_algorithm`, `from_torch`, `from_torch_adapter`,
`from_joblib_estimator_recipe`, `from_parallel_ensemble`, and
`from_iterative_optimization`. Direct `AlgorithmRegistration` construction is
allowed only when the same public invariants and conformance checks are
preserved.

Do not self-award support. `tested`, `supported`, and
`validated_execution_profiles` are evidence-governed fields. Local unit tests,
descriptor discovery, a Wheel build, or a passing mock do not prove a real
cluster profile. Only profiles and semantics covered by the corresponding
fixed-version Gate may be described as validated or supported.

When adding an algorithm or changing a `tributo.algorithms` entry point, update
together:

- the root `pyproject.toml` entry point and owning optional extra;
- the descriptor and all four relevant contract bindings;
- the root `uv.lock` when dependency resolution changes;
- package exports and focused tests;
- `README.md` and `tests/test_package_conformance.py` when the package map or
  official entry-point count changes.

These synchronized updates apply to a `tributo.algorithms` algorithm entry
point. A change limited to an exporter, validator, or model-flavor entry point
must update that plugin's metadata and focused tests without needlessly
changing the algorithm's four Contract Bindings.

Cross-domain composition is permitted only through an intentional import of a
stable public symbol and a complete dependency closure in the consumer's
optional extra. The causal packages reuse shared causal/XGBoost behavior; do
not expand private cross-domain imports or omit required external dependencies
from the consumer's extra. If a shared surface is needed, either use an
existing exported public symbol or propose a versioned public surface
explicitly.

## Data, execution, and artifact boundaries

Algorithm packages consume Core-provided canonical Ray Data inputs and typed
input bindings. They must not implement file scanning, SQL/database readers,
object-store transport, fragment/transaction writers, or a second data
gateway. Domain conversion is allowed for the declared shape of the
algorithm: dense tabular tensors, fixed windows, tokens, graph roles, jagged
histories, or causal treatment/fold roles.

Training input must be lossless for the declared coverage contract. Uneven
shards, empty partitions, insufficient rows, duplicate roles, missing labels,
and invalid feature order must be rejected or handled by an explicitly tested
protocol. The current XGBoost path contains
`CompleteCoverageDataConfig` to preserve remainder rows in Ray Train; retain
that coverage guarantee and do not introduce another general-purpose data
policy for new algorithms. That compatibility module mirrors Ray's
`DataConfig.configure` behavior and uses its protected configuration fields;
it imports `ExecutionResources` through the public `ray.data` package. This is
a known compatibility boundary, not an approved pattern for new code; changes
should move the behavior to a supported public or Core-owned mechanism before
claiming a new support profile.

The migrated Graph PyG, multistage Torch, and jagged recommendation algorithms
use the Core `RAY_TRAIN_TORCH` role-routing contract and still require their
fixed-Wheel exact-coverage Gate before being described as validated. DoWhy/GCM
remain Framework Native. DR and X-Learner build stratified lossless folds, but
currently materialize filtered datasets in the driver; that is a separate
scalability limitation and must not be generalized as the Framework Native
input pattern.

Daft is a data read/write and ETL concern only. Official algorithm training
and formal inference use the Ray path. An algorithm package must not add a
Daft executor, Daft UDF inference, or implicit Ray/Daft engine switching.

Successful formal training is Bundle-backed. Keep these boundaries separate:

- a Ray/framework checkpoint is recovery input to the export path;
- an `ArtifactDraft` is staged output from an exporter;
- a published Bundle is the immutable, digest-checked consumption artifact.

An exporter must declare and validate its source kind, format, flavor, typed
input/output signature, feature order, dtype/shape, and required metadata.
Native formats, Safetensors, PT2, report artifacts, or framework-specific
models need an explicitly registered validator, flavor/loader, and compatible
runtime before they can be treated as serving-capable. Never choose a loader
from a file extension or format string alone. Keep credentials and secret
values out of configs, logs, exceptions, Bundle metadata, and model
references.

Multi-stage algorithms must declare stable `component_stages` and emit
evidence for every stage. Cross-fitting, refutation, nuisance fitting, graph
attribution, distillation, and pretraining stages are not proof of one model's
distributed training unless the receipt also proves the declared model/state
contribution. Cross-Job durable StageGraph scheduling is not provided by this
repository.

## Dependency and package rules

The repository is one uv project built with Hatchling. `packages/*` contains
source domains, while the root `pyproject.toml` owns distribution metadata,
entry points, optional dependency extras, and development configuration. It
targets Python `>=3.12,<3.14` and depends on the published Tributo range
`>=1.1`; it does not use a local Tributo source override.

The graph-pyg and all extras constrain Torch to `>=2.5,<2.14` while PyG's
module imports still call TorchScript APIs deprecated in Torch 2.14. Custom
ONNX graph builders use `make_model_gen_version` to derive the minimum IR
required by their declared opsets; do not inherit the installed ONNX package's
latest IR when the graph does not use its features.

On macOS arm64 with Python 3.12, the uv source configuration selects the
official SciPy 1.15.3 macOS 12 Wheel. DoWhy 0.14 caps SciPy at 1.15.3 on that
Python version, and the macOS 14 Wheel fails dyld validation on affected
systems. Preserve the platform marker and official Wheel hash when updating
the lock. These uv sources are development configuration; pip consumers follow
the platform-specific installation instructions in `README.md`.

Keep these dependency layers distinct:

- package metadata declares the public compatible range;
- `uv.lock` records the reproducible monorepo resolution;
- a production Ray image or offline Wheelhouse supplies the fixed runtime
  closure used by a real Gate.

Do not pin or repackage Tributo's Ray dependency independently in the algorithm
Wheel. Align descriptor `EnvironmentSpec.dependencies`, the root optional
extras, lock resolution, and the tested Image Profile. Keep heavy or
ABI-sensitive stacks such as Torch/PyG, Transformers, DoWhy, causal-learn, and
XGBoost in their domain extras. Consider a separate distribution only when a
demonstrated dependency, ABI, checkpoint, loader, security, support, or release
constraint cannot safely be represented by an optional extra.

All persisted configuration and examples use JSON-oriented contracts. Do not
document YAML as supported by the Tributo CLI, and do not broaden a package's
configuration surface by silently accepting unknown keys.

The release workflow accepts one `tributo-algorithms-v<semver>` tag and builds
one Wheel and source distribution. It publishes the same checksummed files to
TestPyPI, installs and verifies the published distribution, then uploads those
files to PyPI. The tag version, root project version, descriptor
`package_version`, and reviewed artifacts must agree. The
`AlgorithmSpec.version`, implementation version, and descriptor API version
are separate identities and must not be forced to equal the package version.
A tag push and PyPI publication are external operations and require separate
approval. CI runs pull-request validation and push validation on `master`,
using the published Tributo dependency rather than a sibling checkout.

## Build and validation

Use the locked workspace environment for the normal package gate:

```bash
uv sync --locked --extra all --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy packages
uv run pytest tests
uv build --no-sources
```

The tests currently cover descriptor metadata, contract validation, algorithm
mathematics, stage/coverage invariants, exporter/flavor behavior, and package
conformance. `tests/test_package_conformance.py` verifies all 37 official algorithm entry
points from the single installed distribution through Tributo's
descriptor-only conformance testkit, requires four contract bindings per
descriptor, and checks that discovery does not import the selected heavy
implementation frameworks. Preserve this test's subprocess and
count/ownership assertions when changing package composition.

The repository's normal CI runs the algorithm unit, mathematics, contract,
exporter, flavor, and installed-package conformance tests, including the
XGBoost package tests. Tributo Core CI validates only framework contracts with
Core-owned fixture plugins; it does not build or install the official
`tributo-algorithms` distribution. There is currently no real multi-node
Ray Jobs Gate for the aggregate algorithm Wheel in either repository.

Unit tests, local execution, single-process Torch checks, descriptor-only
conformance, and successful Wheel builds do not certify cross-node training.
Do not describe an execution profile as multi-node validated or supported
without evidence from a real fixed-Wheel Ray Jobs run. A future real Gate needs
an explicitly approved owner and implementation plan; it must not make Core CI
depend on the algorithm distribution. Do not use Ray Client as a substitute
for a Ray Data cluster test.

For every algorithm-level change, test the mathematical contract and its
failure boundaries, including as applicable:

- empty and uneven shards, insufficient rows, missing/extra columns, and
  duplicate or incomplete role coverage;
- deterministic seeds, state identity, merge associativity, round barriers,
  stage completeness, and retry safety;
- finite/NaN/Inf inputs, extreme logits or values, invalid hyperparameters,
  class imbalance, overlap requirements, and dtype/shape mismatches;
- checkpoint corruption or incompatible metadata, required artifact failure,
  Bundle digest/signature mismatch, loader rejection, and dynamic batch
  behavior;
- baseline parity with the declared `exact`, `conditional`, or `approximate`
  semantics and tolerance.

When a planned test uses Docker, a database, a Ray cluster, or another remote
service, establish the required test matrix and execution ledger first. Keep
the complete log, code/configuration revision, environment/profile, and result.
Do not repeat a passed long-running suite in the same state without a stated
new reason and the required approval. Never use global Docker prune or broad
cleanup; apply Tributo's precise project/resource cleanup rules.

## Worktree and Git rules

The canonical repository at `~/GitHub/tributo-algorithms/` must remain on its
default `master` branch and clean. Feature code and normal project changes
belong in a dedicated worktree:

```text
~/GitHub/workspace/tributo-algorithms-<descriptive-branch>/
```

Do not use `pr1`, `pr2`, `phase1`, or similar temporary numbering in branch or
worktree names. After creating a worktree, explicitly configure:

```bash
git -C ~/GitHub/workspace/tributo-algorithms-<descriptive-branch> config user.name "jiangxt2"
git -C ~/GitHub/workspace/tributo-algorithms-<descriptive-branch> config user.email "jiangxt2@vip.qq.com"
```

Before modifying files or running project Git operations, verify the actual
repository with `git rev-parse --show-toplevel` and confirm the path is the
planned worktree. The canonical directory is for read-only comparison and
fast-forward synchronization. The initial creation of this project-level
`AGENTS.md` is the narrow, explicitly approved canonical-guide exception; it
does not authorize code changes in the canonical checkout.

Some local environments globally ignore `AGENTS.md`. Check with
`git check-ignore -v AGENTS.md`; do not edit the repository `.gitignore` to
work around that rule. When this guide is intended to be included in an
approved commit, force-add this exact file from the approved worktree and
verify the staged diff; until then, do not describe it as tracked repository
content.

Commits, pushes, releases, PRs, Issues, comments, and external messages each
require separate user approval. Before a commit, inspect every staged file and
verify the required `Signed-off-by: jiangxt2 <jiangxt2@vip.qq.com>` trailer.
Do not publish the algorithm Wheel merely because the local monorepo gate
passes; release the exact aggregate tag, Wheel, dependency closure, and support
evidence that were reviewed together.

## Review checklist

Before declaring a change complete, review the cumulative diff and surrounding
context as a reviewer would:

- every changed descriptor, entry point, package dependency, contract digest,
  validator, exporter, flavor, and public export agrees with the source;
- no new code imports a private Core/Ray module, reads data through a private
  path, bypasses Core lifecycle control, or introduces a second registry; the
  XGBoost `CompleteCoverageDataConfig` compatibility boundary remains
  explicitly identified as migration debt;
- declared distributed semantics match the actual contribution of shards,
  units, rounds, framework stages, and retries;
- config/input/output/coverage contracts test the behavior they claim, not
  merely successful object construction;
- Bundle roles, signatures, digests, loader/flavor identity, and error paths
  are fail-closed and credential-safe;
- current support language distinguishes available, compatible, tested,
  validated, and supported, and does not turn a plan or a local test into a
  production claim;
- package count, entry-point count, README map, lock file, and release tag
  parsing remain consistent;
- Markdown code fences, tables, links, commands, whitespace, and line endings
  are valid, and `git diff --check` is clean.

If a review finds a capability mismatch, fix the source or narrow the
descriptor/support claim. Do not hide it by skipping tests, weakening a
validator, or changing terminology.

## Explicit non-goals

Do not add visual, generative, diffusion, or reinforcement-learning packages
without a separately approved lifecycle, dependency, artifact, and real-Gate
design. Do not reintroduce production algorithms into Tributo Core, restore
the removed `DataConnector` control plane, or make Daft a formal inference
backend. Future algorithms must first be classified by lifecycle, distributed
state semantics, data shape, dependency/ABI closure, checkpoint/Bundle
compatibility, and support profile before choosing an existing package or
creating a new one.
