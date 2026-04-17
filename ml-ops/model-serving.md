# Model Serving

## TL;DR

Model serving means **wrapping your trained AI model in an API so other apps can call it**. You package the model + code into a Docker container, expose it as an HTTP endpoint (usually with FastAPI), and run it somewhere users can reach. Without serving, your model is a script that runs on your laptop — with serving, it's a product.

> 💡 **Key Insight:** A model that can't be called by other code is just a science experiment. Serving is the bridge from "it works on my machine" to "it works for everyone."

---

## The Mental Model

Think of it like a **restaurant kitchen**.

- The model (trained weights) → the chef's recipe/skill
- FastAPI wrapper → the kitchen window where orders come in
- Docker container → the entire restaurant building (kitchen + staff + equipment, all bundled)
- Kubernetes / cloud hosting → the real estate — where the restaurant is located

Mapping:
- Chef's recipe → Trained model weights
- Order ticket → HTTP request (JSON payload)
- Plate of food → HTTP response (model prediction)
- Restaurant chain → Multiple replicas running in parallel

You don't eat in the kitchen. Customers interact through the window. Model serving is building that window.

---

## Why It Exists

### The Problem Before

You trained a model in a Jupyter notebook. It works great. Now what?

```
Before serving:
  data scientist: python predict.py --input "some text"
  result: printed to terminal

  Can your React app call that? No.
  Can 1,000 users use it simultaneously? No.
  Can it run on a server in the cloud? Not easily.
```

### The Solution

Wrap the model in a web server. Now any app — React frontend, mobile app, another microservice — can call it with a simple HTTP request.

```
After serving:
  POST https://api.myapp.com/predict
  Body: { "text": "some text" }
  Response: { "label": "positive", "confidence": 0.94 }

  Any language, any platform, any number of concurrent users.
```

---

## Core Concepts

### 1. FastAPI — The Web Server

**One-line definition:** FastAPI is a Python web framework that turns your model into an HTTP API with almost no boilerplate.

**Analogy:** FastAPI is to Python APIs what Next.js API routes are to TypeScript. You write a function, decorate it with `@app.post("/predict")`, and it's an endpoint. Done.

**Technical explanation:** FastAPI uses Python type hints to automatically validate incoming JSON, serialize outgoing responses, and generate interactive API docs (Swagger UI). It's async-first, so it handles many concurrent requests efficiently.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define what the request body looks like
class PredictRequest(BaseModel):
    text: str

# Define what the response looks like
class PredictResponse(BaseModel):
    label: str
    confidence: float

# Load model once at startup (not on every request!)
model = load_my_model("model.pkl")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    label, confidence = model.predict(request.text)
    return PredictResponse(label=label, confidence=confidence)
```

**Common misconception:** People think FastAPI is just for ML. It's a general-purpose API framework — you use it exactly like Express.js but in Python.

---

### 2. Docker — The Packaging System

**One-line definition:** Docker packages your code, model, dependencies, and OS settings into a single portable unit called an **image** that runs identically everywhere.

**Analogy:** Docker is like an npm package, but instead of just your code, it also packages Node.js itself, the OS, and every system library. Ship the whole environment, not just the code.

```javascript
// Without Docker — "it works on my machine" problem:
// Your laptop: Python 3.11, PyTorch 2.1, CUDA 12.1 → ✅
// Server:      Python 3.9,  PyTorch 1.8, CUDA 11.3 → 💥

// With Docker:
// Your laptop → builds image (frozen snapshot of everything)
// Server      → runs exact same image → ✅ guaranteed to work
```

**Key Docker concepts:**

```
Dockerfile    → Recipe for building the image (like package.json + all setup)
Image         → The built snapshot (like a compiled binary or dist/ folder)
Container     → A running instance of an image (like a running Node.js process)
Registry      → Where images are stored (Docker Hub, AWS ECR = npm registry)
```

**Technical explanation:** Docker uses Linux namespaces and cgroups to create isolated environments. Each container sees its own filesystem, network, and process space. This means you can run 10 different models on one machine without conflicts.

```dockerfile
# Dockerfile for a model serving API

# Start from a base image that has Python + common ML libraries
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caches layers — this speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model
COPY main.py .
COPY model.pkl .

# Tell Docker which port the app listens on
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build the image
docker build -t my-model-api:v1 .

# Run the container locally
docker run -p 8000:8000 my-model-api:v1

# Now test it:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
# {"label": "positive", "confidence": 0.97}
```

**Common misconception:** People think Docker is complicated. The core concept is simple: build a box that contains everything your app needs, ship the box.

---

### 3. Model Optimization — Making It Fast Enough

**One-line definition:** Model optimization reduces the model's size and inference time without significantly hurting accuracy.

**Analogy:** It's like code splitting and tree-shaking in webpack — you're removing the parts you don't need and compressing what's left so it loads faster.

**The main techniques (rough ranges — always benchmark on your own model):**

```
Technique          What It Does                        Typical speedup   Accuracy impact
──────────────────────────────────────────────────────────────────────────────────────
Quantization       Lower-precision weights             1.5-4×            Small (<1-2%)
  (FP32→FP16/INT8/INT4)
Pruning            Zero out low-magnitude weights      1.5-3×            Small to moderate
Knowledge distill. Train a smaller student model       3-10× (smaller)   Moderate
ONNX Runtime       Portable optimized runtime          1.2-2×            None
TensorRT / vLLM    GPU-native inference                2-10×             None to small
Batching           Group many requests per GPU call    2-20× throughput  None
```

> ⚠️ **Numbers vary a lot.** A 7B LLM on an A100 behaves very differently from a BERT classifier on CPU. Use these as "expect *some* of this range" — not as a promise.

```python
# Example: dynamic quantization with PyTorch (CPU, for Linear layers)
import torch
from torch.ao.quantization import quantize_dynamic  # modern API

model = MyModel().eval()

# Quantize Linear layers to INT8 — good for CPU inference on Transformer-style nets
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

# For LLMs, you almost never hand-roll quantization — use GPTQ/AWQ/bitsandbytes
# via libraries (transformers, vLLM, llama.cpp) which handle calibration for you.
```

**Common misconception:** You always need to optimize. Wrong — start with correctness, then measure. For LLMs, the biggest single win is usually **switching from naive HuggingFace+FastAPI to vLLM/TGI** (see §5), not hand-tuning quantization. For classical models, **batching** often beats everything else.

---

### 4. Serving Patterns

**Synchronous (Request-Response):**
```
Client → POST /predict → Server → response in 200ms → Client
Good for: Fast models, interactive features, real-time needs
```

**Asynchronous (Queue-based):**
```
Client → POST /predict → Server → { "job_id": "abc123" }
                                     ↓ (model runs in background)
Client → GET /result/abc123 → { "status": "done", "result": ... }
Good for: Slow models, video processing, batch jobs, long offline LLM tasks
```

**Batch Inference:**
```
Don't run model on each item individually — group items and process together
1000 items × 50ms each = 50 seconds total
vs.
1000 items in 1 batch = 5 seconds total (10× faster)

Good for: Embeddings, offline processing, nightly jobs
```

**Streaming (token-by-token for LLMs):**
```
Client → POST /chat (stream=true) → Server opens SSE / chunked HTTP
                                      ↓
  ← chunk: "The"      ← chunk: " capital"  ← chunk: " of"  ← chunk: " France" ...
                                      ↓
Response completes.  Perceived latency drops dramatically.

Good for: chat UIs, long LLM responses, anywhere time-to-first-token matters
Key metric: TTFT (time-to-first-token), not end-to-end latency
```

---

### 5. Serving LLMs Specifically (vLLM, TGI, SGLang)

**One-line definition:** Dedicated inference servers that keep a GPU continuously fed with tokens from many concurrent requests, instead of one-request-at-a-time.

**Analogy:** A standard FastAPI+PyTorch loop serves LLMs like a single-lane checkout. vLLM serves them like a dozen lanes with a smart manager — partially-done orders get interleaved so no lane sits idle.

**Why FastAPI+PyTorch alone is not enough for LLMs:**

```
Naive LLM serving (PyTorch + FastAPI):
  request A starts → generates 200 tokens → 5s
  request B waits  → generates 200 tokens → another 5s
  GPU utilization: ~30%  (huge idle gaps)

vLLM / TGI / SGLang (continuous batching + PagedAttention):
  A starts, B starts, C starts — all interleaved per-token
  GPU utilization: >90%
  Throughput: 5-20× higher on the same hardware
```

**Key techniques these servers implement:**

- **Continuous batching** — batch new requests in *mid-generation*, not just at the start
- **PagedAttention (vLLM)** — KV-cache is paged like virtual memory, eliminating fragmentation
- **Prefix caching** — common system-prompt prefixes share KV-cache across users
- **Speculative decoding** — a small draft model proposes tokens, the big model verifies in parallel
- **Tensor / pipeline parallelism** — split a big model across multiple GPUs

**The main options:**

| Server | Who / License | Strength |
|--------|---------------|----------|
| **vLLM** | UC Berkeley, Apache 2.0 | Highest throughput, widest model support |
| **TGI** (Text Generation Inference) | Hugging Face, Apache 2.0 | Best integration with HF ecosystem |
| **SGLang** | Open source | Fastest for structured output / complex prompts |
| **TensorRT-LLM** | NVIDIA | Lowest latency on NVIDIA GPUs, harder setup |
| **Ollama / llama.cpp** | Open source | Local/dev, CPU + consumer GPU |

```bash
# Serving Llama 3 with vLLM — one command, OpenAI-compatible API
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000

# Now your code can point the OpenAI SDK at it:
# OpenAI(base_url="http://localhost:8000/v1")
```

**When you don't need this:** If you're calling a hosted API (Anthropic, OpenAI, Bedrock), they handle all of this for you — you just need FastAPI for your *app* logic. You only reach for vLLM/TGI when self-hosting an open-weights LLM.

**Common misconception:** ❌ "I'll just serve Llama with FastAPI + transformers." ✅ That works for demos. At any real traffic, you need vLLM/TGI — the throughput gap is an order of magnitude.

---

## How It Actually Works (Step-by-Step)

Let's trace a single request through a production model serving setup:

```
Step 1: Client sends request
        POST https://api.myapp.com/sentiment
        Body: { "text": "I love this product!" }

Step 2: Load balancer receives request
        Routes to one of 3 running containers (round-robin)

Step 3: FastAPI receives the request
        Validates JSON against PredictRequest schema
        Returns 422 if validation fails (wrong fields, wrong types)

Step 4: Pre-processing
        Tokenize text: "I love this product!" → [101, 1045, 2293, ...]
        Batch with other concurrent requests (micro-batching)

Step 5: Model inference
        Tokens → Neural network → Logits [0.03, 0.97]
        Argmax → class 1 → "positive"

Step 6: Post-processing
        Apply confidence threshold
        Format response: { "label": "positive", "confidence": 0.97 }

Step 7: FastAPI returns JSON response
        200 OK in ~50ms

Step 8: Logging
        Log latency, input hash, output, model version
        (Don't log full inputs — privacy!)
```

---

## Code in Practice

### 1. Hello World — Minimal Sentiment API

```python
# main.py — complete working example
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment API")

# Load once at startup — never inside the endpoint function!
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

class Request(BaseModel):
    text: str

class Response(BaseModel):
    label: str
    score: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Response)
async def predict(req: Request):
    result = classifier(req.text)[0]
    return Response(label=result["label"].lower(), score=result["score"])
```

```bash
# Run locally
pip install fastapi uvicorn transformers torch
uvicorn main:app --reload

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is fantastic!"}'
# {"label":"positive","score":0.9998}
```

### 2. Production Pattern — With Startup/Shutdown and Error Handling

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Lifespan: load model on startup, clean up on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    app.state.model = load_model("./model.pkl")
    logger.info("Model loaded successfully")
    yield  # App runs here
    logger.info("Shutting down, releasing resources")
    del app.state.model

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(req: Request):
    try:
        result = app.state.model.predict(req.text)
        return {"label": result.label, "confidence": result.confidence}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
```

### 3. Dockerized + Deployed to Cloud Run (GCP)

```bash
# 1. Build image
docker build -t sentiment-api:v1 .

# 2. Tag for Google Container Registry
docker tag sentiment-api:v1 gcr.io/my-project/sentiment-api:v1

# 3. Push to registry
docker push gcr.io/my-project/sentiment-api:v1

# 4. Deploy to Cloud Run (serverless — scales to zero, you pay per request)
gcloud run deploy sentiment-api \
  --image gcr.io/my-project/sentiment-api:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2

# Done! Your model is now at https://sentiment-api-xxxxx-uc.a.run.app
```

---

## Gotchas & Pitfalls

```
❌ Loading the model inside the endpoint function
   Every request takes 30 seconds to load the model
✅ Load the model ONCE at startup with lifespan context

❌ No health check endpoint
   Load balancers can't know if your container is healthy
✅ Always add GET /health that returns 200 OK

❌ Logging full user inputs
   Privacy violation — user data ends up in your logs
✅ Log only hashed inputs, metadata, and output categories

❌ No timeout on model inference
   One slow request blocks the worker forever
✅ Cap it at every layer:
   - SDK / HTTP client timeout on any outbound call
   - Uvicorn/Gunicorn worker timeout
   - For async code that awaits I/O: asyncio.wait_for(coro, timeout=10.0)
   - For sync CPU-bound inference: run it in a threadpool and apply the timeout
     there, OR enforce at the worker level — asyncio.wait_for can't actually
     cancel a blocking synchronous call mid-flight.

❌ One giant container with everything
   8GB image, 5-minute startup time
✅ Use slim base images, multi-stage builds, keep images under 2GB

❌ Hardcoding model path in code
   Can't change model without rebuilding image
✅ Use environment variables: MODEL_PATH = os.environ["MODEL_PATH"]

❌ No version tracking
   "Which model is running in production?" — nobody knows
✅ Return model version in every response header or /health endpoint
```

---

## When to Use / When NOT to Use

### Use This When:
- You need to expose a model as an API for other services or frontends
- Multiple teams or languages need to call your model
- You need to scale inference independently from the rest of your app
- You want containerized, reproducible deployments

### Don't Use This When:
- You only need offline/batch predictions → use a script or Airflow job
- Your model is already a cloud service (OpenAI, Anthropic) → just call their API
- You're still in research/experimentation mode → notebooks are fine for now
- The model is tiny and can run client-side → consider TensorFlow.js or ONNX.js

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| RAG | Your vector search + LLM call needs to be served as an API too |
| Vector Databases | Often co-deployed alongside your model serving API |
| CI/CD for ML | Automates the build → test → deploy cycle for new model versions |
| LLM Observability | Traces every request through your serving layer — prompts, tokens, cost |
| Reliability Patterns | Retries, timeouts, and fallbacks live at / in front of the serving layer |
| Safety & Guardrails | Input/output checks wrap each serve call before returning to the user |

---

## Cheat Sheet

```
FastAPI basics:
  @app.post("/predict")    → Define endpoint
  BaseModel (pydantic)     → Request/response validation
  app lifespan             → Load model once at startup
  GET /health              → Always add this

Docker basics:
  docker build -t name .   → Build image
  docker run -p 8000:8000  → Run container locally
  FROM python:3.11-slim     → Use slim base images
  COPY requirements.txt .   → Copy deps before code (caching)

Optimization order:
  1. Make it correct first
  2. Measure where it's slow
  3. Quantize → batch → distill → TensorRT

Remember:
  1. Load model at startup, not per-request
  2. Docker = ship the whole environment, not just the code
  3. Cloud Run = easiest way to deploy: serverless, scales to zero
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why should you load the model outside the endpoint function?**
Loading inside the function means every HTTP request re-loads the model from disk — potentially taking 10-30 seconds each time. Load once at startup, reuse forever.

**Q2: What does Docker actually solve?**
The "works on my machine" problem. It bundles your code, runtime, and all dependencies into an image that runs identically on any machine that has Docker installed.

**Q3: What's the difference between an image and a container?**
Image = the static snapshot (blueprint). Container = a running instance of that image. Many containers can run from the same image simultaneously.

**Q4: What's the first thing to check when a model API is too slow?**
Are you loading the model on every request? If no, measure where time is spent: pre-processing, inference, or post-processing. Optimize the bottleneck, not random parts.

**Q5: When would you use async inference (job queue) instead of sync?**
When the model takes more than ~2-5 seconds to run — long enough that holding an HTTP connection open is impractical. Examples: video processing, large document summarization, fine-tuning jobs.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [FastAPI official docs](https://fastapi.tiangolo.com) | Best docs of any Python framework. The tutorial section takes 2 hours and covers everything you need. |
| [Docker Getting Started](https://docs.docker.com/get-started/) | Official guide — the first 3 parts cover 90% of what you need for ML serving. |
| [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints) | See how the pros do managed model serving — great reference architecture. |
| *Designing Machine Learning Systems* by Chip Huyen | The best book on production ML. Chapter on serving is particularly excellent. |
| [BentoML](https://docs.bentoml.com) | A framework specifically for ML serving — worth knowing as an alternative to hand-rolling FastAPI. |
