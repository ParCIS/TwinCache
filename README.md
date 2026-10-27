# TwinCache

## Installation

Python 3.10 or newer is required. Run installation commands from this source directory. Install FAISS separately for GPU retrieval; do not install CPU and GPU FAISS into the same environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install '.[cpu]'
twincache --help
```

For GPU retrieval, install a GPU-enabled FAISS build compatible with your CUDA and Python environment, then run `pip install .` without the `cpu` extra. The serving process does not require vLLM or PyTorch when using CPU retrieval and remote model services. GPU FAISS is optional; the pip CPU extra cannot provide GPU search.

## Corpus preparation

Supply your own `chunks.jsonl` and `embeddings.npy`. Each JSONL row must have a `text` string, optionally `title` and `chunk_id`. Embedding row `i`, corpus row `i`, and FAISS ID `i` must identify the same chunk. Corpus and queries must use the same embedding model and dimension. Do not mix embeddings from different models.

```bash
twincache-prepare --chunks data/chunks.jsonl \
  --embeddings data/embeddings.npy --output-dir data/index --nlist 1024
```

The tool validates row alignment, L2-normalizes vectors in batches, writes `chunks.jsonl.offsets.npy` beside the corpus, and builds `data/index/vectors.faiss` and `data/index/search.faiss`. The output directory must not already exist. For a small exact-search corpus, omit `--nlist` and use `vectors.faiss` as both index paths. The corpus must contain at least as many chunks as the configured region capacity. The tool consumes existing vectors; it does not download a dataset or run a model. Flat vector storage requires memory proportional to corpus size; choose a suitable index preparation machine.

## Model services

The following optional launch helpers target the tested vLLM 0.15.1 environment. Install `pip install '.[models]'` in a separate GPU model environment. Model assets and their licenses must be obtained separately. The reranker helper is specific to Qwen3's yes/no scoring adaptation and requires the model's correct scoring chat template; it does not use a generative chat endpoint as a reranker.

Run each model service in its own terminal or process supervisor:

```bash
twincache-model embedding --model /path/to/Qwen3-Embedding-0.6B \
  --served-model-name Qwen3-Embedding-0.6B --gpus 0 --port 8000
twincache-model reranker --model /path/to/Qwen3-Reranker-0.6B \
  --served-model-name Qwen3-Reranker-0.6B --gpus 1 --port 8001 \
  --chat-template /path/to/reranker/chat_template.jinja
twincache-model llm --model /path/to/Qwen3-8B \
  --served-model-name Qwen3-8B --gpus 2,3 --port 8002 \
  --tensor-parallel-size 1 --data-parallel-size 2
```

GPU memory utilization defaults to 0.85 and prefix caching is disabled in these helpers. `--dry-run` prints a command without starting a model. The optional two-worker CPU score frontend adapter only supports vLLM 0.15.1 and does not change GPU scheduling; `--score-workers 1` selects the native frontend. Use native vLLM launch commands for other versions. When colocating embedding and a GPU index, leave sufficient GPU memory for FAISS rather than assuming a model utilization of 0.85 fits every corpus.

## Start the service

```bash
twincache --embedding-base-url http://127.0.0.1:8000 \
  --embedding-model Qwen3-Embedding-0.6B \
  --rerank-score-url http://127.0.0.1:8001/score \
  --rerank-model Qwen3-Reranker-0.6B \
  --flat-index-path data/index/vectors.faiss \
  --search-index-path data/index/search.faiss \
  --chunks-path data/chunks.jsonl \
  --enable-llm --llm-base-url http://127.0.0.1:8002 --llm-model Qwen3-8B
```

The cache starts empty and is populated online. By default the release uses compact regions, a tracked cache budget of 45,040,192 bytes (about 42.95 MiB), region capacity 250, cache-selected budget 30, five per-source priority tiers, hybrid reranking, top-3 stability early stopping, and manuscript history decay with gamma 0.95. Exact repeats still rerank selected evidence; they do not reuse a final cached Top5. Background refresh and speculative prefill are off by default.

The public CLI uses `--theta-g 0.88` and `--theta-f 0.90` as explicit Qwen example operating points, not universal calibrated values. Recalibrate these thresholds for different embeddings, rerankers or workloads. Early stopping checks stability of the scored prefix and can change the final Top5 relative to full reranking; disable it with `--disable-early-stop` when an exhaustive candidate reference is required.

The default `--rerank-transport microbatch` retains the bounded cross-request scheduler (32 pairs, four HTTP requests in flight, 2 ms batching wait). For the direct-send path, add `--rerank-transport direct --rerank-direct-batch-size 30`: each logical submission is split into at-most-30-pair HTTP requests, sent concurrently with no application HTTP in-flight cap. A hybrid hit sends the first three tiers, then the remaining two only if needed. Direct transport does not implement cross-request priority arbitration; vLLM schedules submitted work. These transport modes have different queueing behavior and should not be mixed within one performance curve.

For GPU Flat retrieval, add `--retrieval-backend gpu-flat --gpu-id 0`; the search index path is not needed. For GPU IVF, use `--retrieval-backend gpu-ivf` and provide both index paths. The FAISS device ID is local to the serving process's `CUDA_VISIBLE_DEVICES`. You may select the same physical GPU as the embedding service if memory permits.

## API

```bash
curl http://127.0.0.1:9000/health
curl http://127.0.0.1:9000/query -H 'Content-Type: application/json' \
  -d '{"query":"What is a semantic cache?","request_id":"example-1","enable_llm":true}'
```

`POST /query` returns the generated `answer`, final evidence and timing fields. `POST /query/stream` returns newline-delimited JSON stage events, not a live token-by-token chat stream. Generation uses `/v1/completions` with a document-first prompt. For retrieval and reranking only, omit `--enable-llm` and set `enable_llm` to false in requests. Logs go to `runtime/logs`; the buffered writer has bounded queues and exposes dropped events and failures through health accounting. Do not enable durable logging during latency comparisons without reporting its cost.

## Background refresh

Background refresh is opt-in and requires a device-specific rate:

```text
--enable-background-refresh --refresh-rate 0.25
```

This example admits cached-query refresh tasks at a fixed rate in tasks per second. A task retrieves and reranks database Top60 and updates its matched region; it is not simply a replay of foreground cache-selected candidates. The single background worker, deduplication and mature-region gating are preserved. Calibrate the rate under foreground traffic rather than treating the example as a measured safe rate. All cache-owned history is subject to the same memory budget. The budget is tracked cache state, not total process RSS, model weights or index memory.

## Verification

```bash
python -m unittest discover -s tests -t . -v
```

The tests exercise cache LRU and memory accounting, history decay, candidate partitioning, hybrid early stopping, bounded scheduler observations, telemetry and score transport. A synthetic integration test runs cold misses, exact and semantic hits, and fallback with actual small CPU FAISS indexes and mocked HTTP model responses. It needs no GPUs, private files, downloaded models or original experiment directory. This verifies functional packaging, not GPU performance reproducibility.

## Deployment

Use one serving worker: each process owns its own cache, accounting and FAISS resources. Model services may scale independently. Apply access controls before exposing these APIs to an untrusted network; there is no built-in public authentication or tenant isolation. Query text, document evidence and answers may appear in logs; use `--telemetry-mode disabled` where that is inappropriate. Keep corpus and index contents immutable while a process runs, and recreate regions when replacing corpus contents. Warm up representative shapes, record retries and failures, and drain outstanding requests between measured load points.

## License

TwinCache source code is distributed under the [Apache License 2.0](LICENSE). Models, datasets and dependencies retain their respective licenses and are not covered by this source-code grant.

