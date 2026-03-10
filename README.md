# Trackshot

Real-time ball detection and tracking on the edge using a custom-trained YOLOv8n model deployed on an ESP32-S3 microcontroller with a live web dashboard.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── configs            <- Hydra configuration files
│   ├── model          <- Model configurations
│   ├── data           <- Data configurations
│   └── train.yaml     <- Main training configuration
│
├── checkpoints        <- Trained and serialized models or checkpoints
│
├── logs               <- Logs from experiment trackers like TensorBoard or WandB
│
├── notebooks          <- Jupyter notebooks for exploration
│
├── pyproject.toml     <- Project configuration file with dependencies and tool configs
│
├── train.py           <- Main training entry point
│
└── trackshot   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes trackshot a Python module
    │
    ├── data                    
    │   ├── __init__.py
    │   └── dataset.py          <- Data loaders and dataset definitions
    │
    ├── models                  
    │   ├── __init__.py 
    │   └── architecture.py     <- Neural network architecture definitions
    │
    └── utils                   <- Utilities for metrics, logging, etc.
        └── __init__.py
```

--------
