# ISNet: Infrared Small Target Detection Network

This repository contains the Jittor implementation of ISNet, a network designed for infrared small target detection, based on the U-Net structure with TOAA (Trilateral-branch Omic-attention Aggregation) blocks and TFD-inspired (Taylor Finite Difference) edge blocks.

## Project Architecture

The project is organized to promote modularity, readability, and ease of use.

```
ISNET-on-jitter/
├── data/                     # Datasets (e.g., IRSTD-1k, SIRST)
│   ├── IRSTD-1k/
│   │   ├── train/
│   │   └── test/
│   └── sirst-master/
│       └── ...
├── notebooks/                # Jupyter notebooks for exploration and visualization
├── results/                  # Output directory for trained models, logs, and predictions
│   ├── checkpoints/
│   └── logs/
├── src/                      # Source code for the ISNet project
│   ├── isnet/                # Main package for ISNet
│   │   ├── __init__.py
│   │   ├── configs/          # Configuration files (YAML or Python)
│   │   │   └── default_config.yaml
│   │   ├── datasets/         # Data loading and preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── sirst_dataset.py # Dataset class (e.g., SirstDataset)
│   │   │   └── transforms.py    # Custom data augmentations/transformations
│   │   ├── losses/           # Loss function implementations
│   │   │   ├── __init__.py
│   │   │   ├── dice_loss.py
│   │   │   ├── edge_loss.py
│   │   │   └── combined_loss.py
│   │   ├── metrics/          # Evaluation metrics
│   │   │   ├── __init__.py
│   │   │   └── iou_metrics.py   # (e.g., mIoU, PD_FA, ROC)
│   │   ├── models/           # Model definitions
│   │   │   ├── __init__.py
│   │   │   ├── isnet_model.py   # Main ISNet architecture combining components
│   │   │   ├── encoder_decoder.py # Encoder and Decoder backbone
│   │   │   ├── attention_blocks.py# TOAA block implementation
│   │   │   ├── edge_blocks.py     # TFD-inspired Edge block implementation
│   │   │   ├── common_modules.py  # Basic conv blocks, residual blocks, head
│   │   │   └── DCNv2/             # Placeholder for compiled DCNv2 module
│   │   ├── trainer.py        # Training and validation logic
│   │   └── utils/            # Utility functions
│   │       ├── __init__.py
│   │       ├── logger.py
│   │       ├── jittor_utils.py  # Jittor specific utilities
│   │       ├── visualization.py # For drawing masks, overlays
│   │       └── video_processing.py# For video I/O and frame handling
│   ├── scripts/              # Standalone scripts
│   │   ├── train.py          # Main script to start training
│   │   ├── evaluate.py       # Script for evaluating a trained model
│   │   ├── predict_image.py  # Script for running inference on single images/directories
│   │   └── predict_video.py  # Script for real-time inference on video streams/files
│   └── tests/                  # Unit tests (optional, but recommended)
├── .gitignore
├── LICENSE
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Mapping Paper Architecture to Code Modules

The ISNet architecture, as shown in the paper, can be mapped to the following modules within `src/isnet/models/`:

1.  **Input Processing**:
    *   The Sobel filter for generating the `coarse edge` would typically be part of the data preprocessing pipeline in `src/isnet/datasets/sirst_dataset.py` or `src/isnet/datasets/transforms.py`.

2.  **Main U-Net Structure (`isnet_model.py`)**:
    *   This file will define the overall ISNet class, orchestrating the encoder, decoder, and edge pathway. It will manage the skip connections and the final combination of features.

3.  **Encoder (`encoder_decoder.py`, `common_modules.py`)**:
    *   **Stem block**: A specific convolutional block at the beginning of the encoder.
    *   **Residual blocks**: Standard residual blocks used in the encoder.

4.  **Decoder (`encoder_decoder.py`, `attention_blocks.py`, `common_modules.py`)**:
    *   **Deconvolution & Residual blocks**: Upsampling layers combined with residual blocks.
    *   **TOAA Blocks**: Implemented in `attention_blocks.py` and integrated into the decoder path.

5.  **Edge Pathway (`edge_blocks.py`)**:
    *   **TFD-inspired Edge Blocks**: These specialized blocks for edge feature refinement will be in `edge_blocks.py`. The pathway takes features from the encoder and decoder.

6.  **Head (`common_modules.py`)**:
    *   The final prediction layer(s) that produce the `output` segmentation map.

7.  **Loss Functions (`src/isnet/losses/`)**:
    *   **DiceLoss**: Applied to the main segmentation `output`. Implemented in `dice_loss.py`.
    *   **EdgeLoss**: Applied to the `fine edge` map produced by the edge pathway. Implemented in `edge_loss.py`.
    *   A combined loss might be used in `combined_loss.py` to train the network end-to-end.

8.  **DCNv2**:
    *   As per original instructions, this is an external dependency. The compiled module should be placed in `src/isnet/models/DCNv2/` to be accessible by the model components that require it (likely within `common_modules.py` or specific blocks like TOAA).

## Key Scripts

*   **`src/scripts/train.py`**:
    *   Parses arguments (or loads from `src/isnet/configs/`).
    *   Initializes `SirstDataset` from `src/isnet/datasets/`.
    *   Initializes the `ISNet` model (implemented in Jittor) from `src/isnet/models/`.
    *   Sets up optimizers, learning rate schedulers, and loss functions (Jittor-based) from `src/isnet/losses/`.
    *   Uses the `Trainer` class (adapted for Jittor) from `src/isnet/trainer.py` to manage the training and validation loops.
    *   Saves checkpoints to `results/checkpoints/` and logs to `results/logs/`.
*   **`src/scripts/evaluate.py`**:
    *   Loads a trained Jittor model checkpoint.
    *   Evaluates performance on a test set using metrics (adapted for Jittor) from `src/isnet/metrics/`.
*   **`src/scripts/predict_image.py`**:
    *   Loads a trained Jittor model.
    *   Runs inference on specified input images or a directory of images.
    *   Saves or displays the output segmentation masks.
*   **`src/scripts/predict_video.py`**:
    *   Loads a trained Jittor model.
    *   Performs real-time inference on a video file or a connected camera stream.
    *   Displays the video with segmentation masks overlaid in real-time.

## Setup and Usage

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd ISNET-on-jitter
    ```

2.  **Create and activate a Python environment** (e.g., using conda or venv):
    ```bash
    conda create -n isnet_env python=3.8
    conda activate isnet_env
    ```

3.  **Install Jittor and dependencies**:
    *   Follow the official Jittor installation guide: [https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/install.html](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/install.html)
    *   Install other dependencies (ensure `requirements.txt` is updated for Jittor):
        ```bash
        pip install -r requirements.txt
        ```

4.  **Install DCNv2 (Jittor Version)**:
    *   You will need a Jittor-compatible version of DCNv2. The PyTorch version will not work.
    *   Search for a Jittor implementation of DCNv2 (e.g., on GitHub or Jittor's model zoo if available).
    *   Place the compiled Jittor DCNv2 module into `src/isnet/models/DCNv2/` and ensure it can be imported by your Jittor model code.

5.  **Prepare Datasets**:
    *   Download IRSTD-1k and/or SIRST datasets.
    *   Organize them under the `data/` directory as shown in the structure above. Update paths in dataset configuration or scripts if necessary.

6.  **Configuration**:
    *   Modify configuration files in `src/isnet/configs/` (e.g., `default_config.yaml`) to set hyperparameters, dataset paths, etc.

7.  **Training**:
    ```bash
    python src/scripts/train.py --config src/isnet/configs/default_config.yaml  # Ensure this script uses Jittor
    ```

8.  **Evaluation**:
    ```bash
    python src/scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth --config src/isnet/configs/default_config.yaml  # Ensure this script uses Jittor
    ```

9.  **Prediction on Images**:
    ```bash
    python src/scripts/predict_image.py --checkpoint results/checkpoints/best_model.pth --input /path/to/your/images_or_image.jpg --output /path/to/save/results  # Ensure this script uses Jittor
    ```

10. **Real-time Prediction on Video Stream/File**:
    *   Using a video file:
        ```bash
        python src/scripts/predict_video.py --checkpoint results/checkpoints/best_model.pth --input /path/to/your/video.mp4  # Ensure this script uses Jittor
        ```
    *   Using a camera (e.g., camera ID 0):
        ```bash
        python src/scripts/predict_video.py --checkpoint results/checkpoints/best_model.pth --input 0  # Ensure this script uses Jittor
        ```

## Dependencies

*   Python 3.x
*   Jittor (refer to official documentation for version)
*   NumPy
*   OpenCV-Python (cv2)
*   TensorBoard (Jittor provides `jittor.utils.summary_writer` for TensorBoard compatibility)
*   (Other dependencies as listed in an updated `requirements.txt` for Jittor)

---

This `README.md` provides a comprehensive, standardized structure. You would then need to refactor your existing code from the `old/` directory into this new structure.
