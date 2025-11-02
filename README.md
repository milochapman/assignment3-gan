
# GAN API (Assignment 3)

### Run all commands from the repository root (where app/ and helper_lib/ live).


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
## Theory Questions
Full derivations are in gan_theory_answers.md.
Short answers for the assignment questions:

Transposed convolution with input 8×8, kernel 4, stride 2, padding 1, output padding 1 → output 17×17.

If we only increase the stride from 2 to 3, the output size increases by (input_size − 1).

General formula for transposed conv:
O = (I − 1)·S − 2P + K + OP

To upsample 16 → 32 with no padding/output padding, one valid setup is stride = 2, kernel = 2.

BatchNorm on [6, 8, 10, 6] (no learnable params) → [-0.904, 0.301, 1.507, -0.904].

ReLU sets negative inputs to 0; LeakyReLU keeps a small slope on negatives.

LeakyReLU is preferred in the GAN discriminator to avoid dead units and to keep gradients during training.


## Notes
The virtual environment (.venv/) and dataset folder (data/) are not committed to GitHub.

The project was tested locally with uvicorn and with curl requests.

This README is written to make it easy for the instructor to pull the repo and run it.
```bash
::contentReference[oaicite:0]{index=0}
```

