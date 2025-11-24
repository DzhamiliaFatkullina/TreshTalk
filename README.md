# TrashTalk: Intelligent Waste Detection and Classification System

TrashTalk is a comprehensive computer vision system designed for automatic detection and classification of waste objects in images. The project combines state-of-the-art object detection with deep learning classification to identify various types of recyclable and non-recyclable materials with high accuracy.

## Project Overview

TrashTalk helps automate waste sorting by identifying different types of trash in images. It can detect multiple waste items in a single photo and classify them into categories like plastic, paper, metal, glass, and special items like batteries.

The system uses proven AI models that have been trained on many waste images, making it accurate and reliable for real-world use. Whether you're taking photos of recycling bins or individual items, TrashTalk can quickly tell you what type of waste each object is.

## Key Features

- **Multi-stage Pipeline**: Implements a sequential workflow of object detection followed by fine-grained classification
- **Multiple Model Support**: Includes various CNN architectures (ResNet, EfficientNet, MobileNet, ConvNeXt) for flexible deployment
- **Comprehensive Dataset**: Unified dataset from multiple public sources with extensive quality analysis
- **Quality Assessment**: Built-in image quality evaluation for preprocessing optimization
- **Modular Design**: Clean separation of components for easy maintenance and extension

## System Architecture

The pipeline follows a structured approach:

1. **Object Detection**: YOLO-v8  identifies potential waste objects in input images
2. **Region Extraction**: Bounding boxes are used to crop detected objects
3. **Classification**: Pre-trained models classify each crop into specific waste categories
4. **Results Aggregation**: Combines detection and classification results with confidence scores

## Supported Waste Categories

The system classifies objects into 9 unified categories:
- Cardboard
- Paper  
- Plastic
- Metal
- Glass
- General Trash
- Batteries
- Clothing/Textiles
- Biological Waste

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- OpenCV
- Kaggle API (for dataset download)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/meldilen/TreshTalk.git
cd trashtalk

# Install dependencies
pip install -r requirements.txt

# Download and prepare datasets
python src/data/download_kaggle.py

# Unify datasets with quality analysis
python src/data/unify.py

# Train model comparison
python src/models/train_comparison.py

# Run inference
python src/pipeline/pipeline.py
```

## Model Training

The project includes a comprehensive model comparison framework that evaluates multiple architectures:

```python
from src.models.train_comparison import run_model_comparison

# Compare all supported models
results, num_classes = run_model_comparison()
```

Supported models include ResNet-18/50, MobileNetV3, EfficientNet-B0/B2, and ConvNeXt-Tiny with automatic selection of the best performing architecture based on validation accuracy and model size.

## Dataset Management

The system automatically handles dataset acquisition and unification from multiple sources:

- **Automated Download**: Fetches datasets from Kaggle using the official API
- **Intelligent Unification**: Maps diverse labeling schemes to unified categories
- **Quality Analysis**: Performs comprehensive image quality assessment
- **Stratified Splitting**: Ensures balanced train/validation/test splits

## Usage Examples

### Basic Inference

```python
from src.pipeline.pipeline import create_complete_pipeline

# Initialize the complete pipeline
pipeline = create_complete_pipeline()

# Process an image
results = pipeline.process_image("path/to/image.jpg")

# Visualize results
pipeline.visualize_results("path/to/image.jpg", results)
```

### Custom Training

```python
from src.models.baselines import MODEL_BUILDERS

# Initialize a custom model
model = MODEL_BUILDERS['efficientnet_b2'](num_classes=10, pretrained=True)
```

## Project Structure

```
TrashTalk/
├── src/
│   ├── data/           # Dataset download and unification
│   ├── models/         # Model architectures and training
│   ├── detection/      # Object detection components
│   └── pipeline.py       # End-to-end pipeline
├── data/
│   ├── raw/           # Original datasets
│   └── unified/       # Processed unified dataset
└── reports/           # Training results and comparisons
```

## Performance

The system achieves strong performance across multiple metrics:
- High classification accuracy on diverse waste types
- Robust detection of multiple objects per image
- Efficient inference suitable for real-time applications
- Comprehensive quality assessment for input validation

## Contributing

We welcome contributions to improve TrashTalk:
- Report bugs and issues
- Suggest new features or improvements
- Add support for additional waste categories

- Improve model performance and efficiency
