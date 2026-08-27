# Tributo Official Algorithms

This monorepo contains official algorithm implementations for Tributo. Each
domain package is an independently buildable, testable, versioned, and
publishable Wheel. Third-party algorithms use the same public contracts and
entry-point path.

Tributo Core owns Ray execution, resource allocation, retries, input leases,
checkpoint transport, evidence, Tune, Bundle publication, and inference.
These packages own model mathematics, framework-native hooks, executable
contracts, and algorithm-specific delivery plugins.

| Package | Algorithms and roles |
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

## Development

```bash
uv sync --all-packages --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy packages
uv run pytest tests
uv build --all-packages
```

Package tags follow `<directory>-v<semver>`, such as `classical-v1.2.0`, so
one package can be released without publishing unrelated Wheels.
