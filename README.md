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

The distribution, domain extras, and algorithm descriptors require Tributo Core
`>=1.1` without an upper version bound. Versions below 1.1 are outside the
declared compatibility range. An open upper bound does not certify future Core
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
