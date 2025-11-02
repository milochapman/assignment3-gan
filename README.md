
# GAN API (Assignment 3)

## Train
```bash
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
pip install "numpy<2" --force-reinstall             # macOS PyTorch/NumPy fix
python train_gan_mnist.py
```

## Run API
```bash
uvicorn app.main:app --reload --port 8000
# docs:   http://127.0.0.1:8000/docs
# health: http://127.0.0.1:8000/health
```

## Generate
```bash
curl -X POST "http://127.0.0.1:8000/generate_with_gan" \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 2}'
```

##Save Returned Images
```bash
curl -s -X POST "http://127.0.0.1:8000/generate_with_gan" \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 4}' > out.json

python - <<'PY'
import json, base64
with open("out.json") as f:
    data = json.load(f)
for i, b in enumerate(data["images"]):
    with open(f"gan_{i}.png", "wb") as f2:
        f2.write(base64.b64decode(b))
print("saved", len(data["images"]), "images")
PY
```

## Docker
```bash
docker build -t gan-api .
docker run --rm -p 8000:8000 gan-api
```
