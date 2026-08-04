# Transformer-Based Multiclass Software Defect Prediction

An AI-powered software defect prediction system leveraging **transformer-based models**,
with **CodeBERT** as the primary model, to classify source code into multiple software
defect categories.

> **Status:** 🚧 Early development — Milestone 0 (project skeleton).
> Environment and structure are established. No model is trained yet.

---

## Problem Statement

Traditional software defect prediction relies on handcrafted software metrics and binary
classification (defective vs. non-defective). These approaches fail to capture the semantics
of source code and cannot identify the *type* of defect present.

This project analyses source code directly using transformer models to perform **multiclass
software defect prediction**, helping developers identify specific defect categories.

## Objectives

- Develop a transformer-based multiclass defect prediction framework
- Fine-tune CodeBERT for defect-type classification
- Compare against alternative code encoders
- Provide explainable, token-level defect analysis
- Serve predictions through a working web interface

## Planned Approach

| Component | Technology |
|---|---|
| Dataset | IBM Project CodeNet (C++ submissions) |
| Primary model | CodeBERT |
| Comparison models | UniXcoder, GraphCodeBERT, RoBERTa *(planned)* |
| Backend | FastAPI |
| Frontend | Streamlit |
| Framework | PyTorch + Hugging Face Transformers |
| Headline metric | Macro-F1 (chosen for class imbalance) |

## Project Structure

Current state of the repository:

```text
transformer-defect-prediction/
├── app/            # Streamlit frontend
├── configs/        # YAML configuration
├── docs/           # Design documentation
├── src/sdp/        # Main package
│   ├── api/        # FastAPI service
│   └── model/      # Model implementations
├── tests/          # Test suite
├── pyproject.toml
└── requirements.txt
```

Additional modules (`data/`, `training/`, `evaluation/`) will be added as milestones complete.

## Setup

Requires **Python 3.12**.

```bash
git clone https://github.com/<your-username>/transformer-defect-prediction.git
cd transformer-defect-prediction

# Windows
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate

# PyTorch is installed CPU-only; training runs on Colab GPU
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Roadmap

| | Milestone | Status |
|---|---|---|
| M0 | Project skeleton + walking skeleton | 🚧 In progress |
| M1 | Dataset acquisition, labelling, splits | ⏳ |
| M2 | CodeBERT baseline + evaluation | ⏳ |
| M4 | Label taxonomy enrichment | ⏳ |
| M5 | Hierarchical head + imbalance handling | ⏳ |
| M6 | Model comparison study | ⏳ |
| M8 | Explainability | ⏳ |
| M9 | Full evaluation + report | ⏳ |

## Current Limitations

Implemented: project structure, environment configuration, dependency management, tooling.

**Not yet implemented:** dataset acquisition, preprocessing, CodeBERT integration, training,
evaluation, FastAPI backend, Streamlit frontend, explainability. Any prediction returned by
the current codebase is random placeholder output.

## Team

**BE Final Year Project · Group 42**
Department of Computer Engineering, Sinhgad Institute of Technology
Savitribai Phule Pune University · AY 2026–27

**Guide:** Prof. V. M. Chavan

**Members:** Prashant More · Swagat Tonage · Omkar Wadle · Vidya Shinde

## License

MIT — see [LICENSE](LICENSE).
