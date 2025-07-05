# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive data science toolkit for exploratory data analysis, featuring specialized modules for tabular data, text/NLP analysis, and image processing. The project is structured as a Python package using Poetry for dependency management.

## Core Architecture

### Main Modules

The toolkit is organized into four primary modules in the `src/` directory:

- **`data_tools.py`** - Core tabular data analysis toolkit with functions for data discovery, cleaning, exploration, grouping, feature engineering, and modeling preparation
- **`nlp_tools.py`** - Natural language processing toolkit for document loading, text preprocessing, entity extraction, feature extraction, and text classification
- **`photo_tools.py`** - Image processing and analysis toolkit for image loading, visualization, preprocessing, and dataset exploration
- **`generate_test_data.py`** - Synthetic data generator for creating realistic test datasets with various data types and edge cases
- **`generate_nlp_data.py`** - NLP data generation utilities for creating synthetic emails and text documents

### Key Features by Module

#### data_tools.py
- Comprehensive EDA functions (`quick_look`, `profile_columns`, `explore_relationships`)
- Advanced grouping and segmentation analysis (`group_analyze`, `multi_group_analyze`, `segment_profiler`)
- Quantile analysis and signal strength detection
- Feature preparation for machine learning
- Interactive dashboards and visualizations

#### nlp_tools.py
- Document loading from multiple formats (txt, csv, json, md)
- Email parsing and metadata extraction
- Text preprocessing with customizable options
- Named entity extraction and n-gram analysis
- TF-IDF vectorization and document classification
- Sentiment analysis and text visualization

#### photo_tools.py
- Image dataset loading and metadata extraction
- Interactive image browser with Jupyter widgets
- Image preprocessing pipeline visualization
- Synthetic image generation with different patterns
- Statistical analysis of image properties

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell

# Alternative: use existing venv
source venv/bin/activate  # On macOS/Linux
```

### Running the Toolkit
```bash
# Launch Jupyter Lab for interactive analysis
poetry run jupyter lab

# Start Jupyter Notebook
poetry run jupyter notebook

# Run Python scripts directly
poetry run python src/data_tools.py
poetry run python src/generate_test_data.py
```

### Working with Notebooks
The `notebooks/` directory contains several analysis notebooks:
- `tabular_data_explore.ipynb` - Tabular data analysis examples
- `nlp_data_explorer.ipynb` - Text processing and NLP analysis
- `photo_data_explorer.ipynb` - Image analysis and computer vision
- `base_photo_modelling.ipynb` - Photo modeling techniques

### Data Management
```bash
# Generate test datasets
poetry run python src/generate_test_data.py
poetry run python src/generate_nlp_data.py

# Data is stored in the data/ directory:
# - data/raw/ - Raw input data
# - data/processed/ - Processed datasets
# - data/test_images/ - Generated test images
```

## Project Structure Notes

### Notebooks Organization
- `notebooks/data/` contains subdirectories for different data types:
  - `documents/` - Text documents for NLP analysis
  - `emails/` - Email data for processing
  - `text_classification/` - Labeled text data
- `notebooks/custom_emails/` and `notebooks/custom_documents/` contain generated synthetic data

### Key Dependencies
- **Core:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **NLP:** nltk, textblob, wordcloud
- **Vision:** opencv-python, PIL/Pillow
- **Interactive:** jupyterlab, ipywidgets
- **Utilities:** tabulate, dvc (data version control)

## Usage Patterns

### Tabular Data Analysis
```python
from src.data_tools import *
df = pd.read_csv('data/your_dataset.csv')

# Quick overview
quick_look(df)

# Complete EDA pipeline
df_clean = perform_eda(df, target_col='target')

# Group analysis
group_analyze(df, 'category', 'target')
```

### NLP Analysis
```python
from src.nlp_tools import *

# Load documents
docs_df = load_text_documents('notebooks/data/documents/')

# Preprocess text
docs_df = add_preprocessed_text(docs_df)

# Extract features and classify
feature_df, vectorizer = extract_document_features(docs_df)
```

### Image Analysis
```python
from src.photo_tools import *

# Load image metadata
img_df = load_image_info('data/test_images/')

# Create interactive browser
browser = ImageBrowser(img_df)
browser.display()

# Analyze with data explorer
explorer = ImageDataExplorer(img_df)
explorer.analyze_image_properties()
```

## Testing and Validation

The toolkit includes comprehensive test data generation:
- `generate_test_data()` creates realistic datasets with various data types, missing values, outliers, and correlations
- `generate_synthetic_emails()` creates email datasets for NLP testing
- `generate_test_images()` creates synthetic images with different patterns for computer vision tasks

## Common Workflows

1. **Initial Data Exploration:** Use `quick_look()` and `profile_columns()` for rapid data assessment
2. **Deep Dive Analysis:** Apply `perform_eda()` for comprehensive analysis with visualizations
3. **Segment Analysis:** Use `group_analyze()` and `segment_profiler()` for detailed breakdowns
4. **Feature Engineering:** Apply `create_quantile_bins()` and `find_signal_strength()` for modeling preparation
5. **Text Processing:** Load documents with `load_text_documents()` and process with `add_preprocessed_text()`
6. **Image Analysis:** Use `ImageDataExplorer` for comprehensive image dataset analysis