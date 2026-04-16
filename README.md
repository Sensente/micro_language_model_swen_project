# Micro Language Models Enable Instant Responses
This is the official repo of our paper **Micro Language Models Enable Instant Responses**.

For more information, please refer to our [paper]().

# Demo

Our demo showing the local `micro lm (Swen)` generated the first `8` words, then `gpt-4o` followed the prefix to finish the answer.

<video src="./assests/figures/demo.mp4" controls width="800"></video>

# Model details

The model shown in the demo is our `28M` version model with 512 `hidden_size` and 8 `layers`.

## Checkpoint and tokenizer

The model checkpoint is available [here](./models/swen_28m.pth).

Tokenizer files are available [here](./models/tokenizer).

# Run the demo

## Setup

Create and activate your environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Copy env template:

```bash
cp .env.example .env
```

Then edit `.env` and fill in:

- `SWEN_CKPT`
- `SWEN_TOKENIZER`
- `OPENAI_API_KEY`

Load environment variables:

```bash
export $(grep -v '^#' .env | xargs)
```

## Run backend

From the repo root where `model/model_swen.py` is importable:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

## Run frontend

In another terminal:

```bash
cd frontend
python -m http.server 3000
```

Open:

```text
http://127.0.0.1:3000
```

## How the continuation works

The backend first runs local decoding until it reaches at least 8 words. It then:

- shows that prefix immediately in the UI
- asks the model on the cloud to **continue after that prefix without repeating it**


# Citation
If you use this codebase, or our idea inspires your work, welcome cite:

```bibtex
@inproceedings{
}