#!/usr/bin/env bash
# Proposed serving/throughput template for future benchmarks (baseline vs GPTQ).
# Requires vLLM on PATH. Replace MODEL_PATH with your local checkpoint.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/llama3-8b-instruct}"
QUANT_KIND="${QUANT_KIND:-gptqmodel}"   # gptqmodel | none
MAX_LEN="${MAX_LEN:-8192}"
TP="${TP:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
HOST="0.0.0.0"
PORT="${PORT:-8000}"

echo "[INFO] Serving $MODEL_PATH quantization=$QUANT_KIND"
vllm serve "$MODEL_PATH"   --quantization "$QUANT_KIND"   --max-model-len "$MAX_LEN"   --tensor-parallel-size "$TP"   --host "$HOST" --port "$PORT"   --swap-space 8   --disable-log-requests &

SERVER_PID=$!
trap "kill $SERVER_PID || true" EXIT

sleep 10

echo "[INFO] Running a tiny concurrent load (approximate latency/TTFT sanity check)..."
python - <<'PY'
import time, requests, threading, os, json
N = int(os.environ.get("CONCURRENCY", "8"))
HOST = os.environ.get("HOST","0.0.0.0")
PORT = os.environ.get("PORT","8000")
# Adjust the endpoint to match your vLLM setup.
URL = f"http://{HOST}:{PORT}/v1/completions"
HEADERS = {"Content-Type":"application/json"}
PROMPT = "Briefly explain why 4-bit weight-only quantization can speed up LLM inference."

times = []
def work():
    t0 = time.time()
    data = {"model":"placeholder","prompt":PROMPT,"max_tokens":128}
    try:
        r = requests.post(URL, headers=HEADERS, data=json.dumps(data), timeout=60)
        r.raise_for_status()
    except Exception as e:
        print("request failed:", e)
        return
    t1 = time.time()
    times.append(t1-t0)
    print("latency:", t1-t0)

threads = [threading.Thread(target=work) for _ in range(N)]
[t.start() for t in threads]
[t.join() for t in threads]
if times:
    print("avg latency:", sum(times)/len(times))
else:
    print("no successful requests")
PY

echo "[INFO] Remember: use the serving engine's metrics for tokens/s; this is just a sanity check."
