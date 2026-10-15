# Trackshot

Real-time ball detection and tracking on the edge using a custom-trained YOLOv8n model deployed on an ESP32-S3 microcontroller with a live web dashboard.

## Web Dashboard (UI)

The live dashboard lives in its own repository, [local-image-capture](https://github.com/pierGit7/local-image-capture), and is included here as a git submodule at `modules/local-image-capture`. It is a Vite + React app that connects to the ESP32-S3 over the browser's Web Serial API and shows the camera stream, detected bounding boxes and telemetry.

### Fetching the submodule

When cloning the project for the first time, fetch the submodule at the same time:

```sh
git clone --recurse-submodules https://github.com/pierGit7/TrackShot.git
```

If you already cloned without `--recurse-submodules` (the `modules/local-image-capture` folder is empty), initialise it:

```sh
git submodule update --init --recursive
```

To pull the latest UI changes from its remote `main` branch:

```sh
git submodule update --remote modules/local-image-capture
git add modules/local-image-capture
git commit -m "Bump UI submodule"
```

### Running the UI

Requires [Node.js](https://nodejs.org/) (v18+) and npm.

```sh
cd modules/local-image-capture
npm install
npm run dev
```

The dev server starts on <http://localhost:8080>. Then:

1. Open the page in **Chrome or Edge** (Web Serial is not supported in Firefox or Safari).
2. Plug in the ESP32-S3 running the Trackshot firmware via USB.
3. Click connect and pick the board's serial port in the browser prompt. The default baud rate is **921600**, which must match the firmware.
4. Close any other serial monitor (e.g. `idf.py monitor`) first, since only one program can hold the port at a time.

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
