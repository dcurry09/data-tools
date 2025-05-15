
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from PIL import Image
import cv2
from IPython.display import display, clear_output
from ipywidgets import widgets, HBox, VBox, Layout
import io
import base64
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ==================== IMAGE LOADING AND BASIC INFO ====================

def load_image_info(image_folder: str, extensions: List[str] = ['jpg', 'jpeg', 'png', 'bmp']) -> pd.DataFrame:
    """
    Load information about all images in a folder
    
    Returns DataFrame with image metadata
    """
    image_folder = Path(image_folder)
    image_files = []
    
    for ext in extensions:
        image_files.extend(image_folder.glob(f'*.{ext}'))
        image_files.extend(image_folder.glob(f'*.{ext.upper()}'))
    
    image_info = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                info = {
                    'filename': img_path.name,
                    'path': str(img_path),
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_kb': img_path.stat().st_size / 1024,
                    'aspect_ratio': img.width / img.height
                }
                image_info.append(info)
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
    
    return pd.DataFrame(image_info)

def quick_image_summary(image_df: pd.DataFrame) -> None:
    """
    Print summary statistics about image dataset
    """
    print("="*50)
    print("IMAGE DATASET SUMMARY")
    print("="*50)
    print(f"Total images: {len(image_df)}")
    print(f"Image formats: {image_df['format'].value_counts().to_dict()}")
    print(f"Color modes: {image_df['mode'].value_counts().to_dict()}")
    print(f"\nSize statistics:")
    print(f"  Width: {image_df['width'].min()} - {image_df['width'].max()} (avg: {image_df['width'].mean():.0f})")
    print(f"  Height: {image_df['height'].min()} - {image_df['height'].max()} (avg: {image_df['height'].mean():.0f})")
    print(f"  File size: {image_df['size_kb'].min():.1f} - {image_df['size_kb'].max():.1f} KB")
    print(f"  Aspect ratios: {image_df['aspect_ratio'].min():.2f} - {image_df['aspect_ratio'].max():.2f}")

# ==================== IMAGE VISUALIZATION ====================

def plot_image_grid(image_paths: List[str], 
                   labels: Optional[List[str]] = None,
                   n_cols: int = 4,
                   fig_width: int = 16,
                   show_info: bool = True) -> None:
    """
    Plot multiple images in a grid
    """
    n_images = len(image_paths)
    n_rows = (n_images - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_width * n_rows / n_cols))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, img_path in enumerate(image_paths):
        ax = axes[idx]
        
        try:
            img = mpimg.imread(img_path)
            ax.imshow(img)
            
            title = Path(img_path).name
            if labels and idx < len(labels):
                title += f"\n{labels[idx]}"
            
            if show_info:
                h, w = img.shape[:2]
                title += f"\n{w}x{h}"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{Path(img_path).name}", 
                   ha='center', va='center')
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ==================== INTERACTIVE IMAGE BROWSER ====================

class ImageBrowser:
    """
    Interactive image browser for notebook environments
    """
    def __init__(self, df: pd.DataFrame, image_col: str = 'path', 
                 metadata_cols: Optional[List[str]] = None):
        """
        Initialize image browser
        
        Parameters:
        - df: DataFrame with image information
        - image_col: Column containing image paths
        - metadata_cols: Columns to display as metadata
        """
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.metadata_cols = metadata_cols or []
        self.current_idx = 0
        
        # Create widgets
        self.setup_widgets()
        
    def setup_widgets(self):
        """Setup interactive widgets"""
        # Navigation buttons
        self.prev_btn = widgets.Button(description='Previous', icon='arrow-left')
        self.next_btn = widgets.Button(description='Next', icon='arrow-right')
        self.idx_input = widgets.IntText(value=0, description='Index:')
        self.total_label = widgets.Label(value=f' / {len(self.df)-1}')
        
        # Metadata display
        self.metadata_output = widgets.Output()
        
        # Image display
        self.image_output = widgets.Output()
        
        # Connect events
        self.prev_btn.on_click(lambda x: self.show_previous())
        self.next_btn.on_click(lambda x: self.show_next())
        self.idx_input.observe(self.on_index_change, names='value')
        
        # Layout
        nav_box = HBox([self.prev_btn, self.idx_input, self.total_label, self.next_btn])
        self.ui = VBox([nav_box, self.image_output, self.metadata_output])
        
    def show_image(self, idx: int):
        """Display image at given index"""
        if 0 <= idx < len(self.df):
            self.current_idx = idx
            row = self.df.iloc[idx]
            
            # Clear previous output
            self.image_output.clear_output(wait=True)
            self.metadata_output.clear_output(wait=True)
            
            # Display image
            with self.image_output:
                try:
                    img_path = row[self.image_col]
                    img = mpimg.imread(img_path)
                    
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.title(f"Image {idx}: {Path(img_path).name}")
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print(f"Error loading image: {e}")
            
            # Display metadata
            with self.metadata_output:
                metadata = {}
                for col in self.metadata_cols:
                    if col in row.index:
                        metadata[col] = row[col]
                
                if metadata:
                    print("Metadata:")
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")
            
            # Update index input
            self.idx_input.value = idx
    
    def show_next(self):
        """Show next image"""
        if self.current_idx < len(self.df) - 1:
            self.show_image(self.current_idx + 1)
    
    def show_previous(self):
        """Show previous image"""
        if self.current_idx > 0:
            self.show_image(self.current_idx - 1)
    
    def on_index_change(self, change):
        """Handle manual index change"""
        new_idx = change['new']
        if 0 <= new_idx < len(self.df):
            self.show_image(new_idx)
    
    def display(self):
        """Display the browser"""
        display(self.ui)
        self.show_image(0)

# ==================== IMAGE WITH METADATA VISUALIZATION ====================

def visualize_image_with_data(image_path: str, metadata: Dict, 
                            target_value: Optional[float] = None,
                            fig_size: Tuple[int, int] = (14, 8)) -> None:
    """
    Display image alongside its metadata and target value
    """
    fig = plt.figure(figsize=fig_size)
    
    # Image subplot
    ax1 = plt.subplot(1, 2, 1)
    img = mpimg.imread(image_path)
    ax1.imshow(img)
    ax1.set_title(f"Image: {Path(image_path).name}")
    ax1.axis('off')
    
    # Metadata subplot
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    # Format metadata text
    text_content = f"Image Information\n{'='*30}\n"
    text_content += f"Dimensions: {img.shape[1]} × {img.shape[0]}\n"
    
    if img.ndim == 3:
        text_content += f"Channels: {img.shape[2]}\n"
    
    text_content += f"\nMetadata\n{'='*30}\n"
    for key, value in metadata.items():
        text_content += f"{key}: {value}\n"
    
    if target_value is not None:
        text_content += f"\nTarget\n{'='*30}\n"
        text_content += f"Value: {target_value}\n"
        
        # Color code based on target
        if isinstance(target_value, (int, float)):
            if target_value > 0.5:
                color = 'green'
                label = 'Positive'
            else:
                color = 'red'
                label = 'Negative'
            text_content += f"Class: {label}\n"
            
            # Add colored border to image
            rect = Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1, 
                           linewidth=5, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
    
    ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

# ==================== IMAGE PREPROCESSING ====================

def analyze_image_stats(image_folder: str, n_samples: int = 100) -> Dict:
    """
    Analyze statistical properties of images for preprocessing
    """
    image_paths = list(Path(image_folder).glob('*.[jp][pn][g]*'))[:n_samples]
    
    pixel_means = []
    pixel_stds = []
    dimensions = []
    
    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate per-channel statistics
            means = [img[:,:,i].mean() for i in range(3)]
            stds = [img[:,:,i].std() for i in range(3)]
            
            pixel_means.append(means)
            pixel_stds.append(stds)
            dimensions.append(img.shape[:2])
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Calculate overall statistics
    pixel_means = np.array(pixel_means)
    pixel_stds = np.array(pixel_stds)
    dimensions = np.array(dimensions)
    
    stats = {
        'mean_per_channel': pixel_means.mean(axis=0),
        'std_per_channel': pixel_stds.mean(axis=0),
        'overall_mean': pixel_means.mean(),
        'overall_std': pixel_stds.mean(),
        'common_dimensions': pd.DataFrame(dimensions, columns=['height', 'width']).mode().iloc[0].to_dict(),
        'dimension_stats': {
            'min_height': dimensions[:, 0].min(),
            'max_height': dimensions[:, 0].max(),
            'min_width': dimensions[:, 1].min(),
            'max_width': dimensions[:, 1].max(),
        }
    }
    
    return stats

def plot_image_preprocessing_pipeline(image_path: str, 
                                    target_size: Tuple[int, int] = (224, 224),
                                    normalize: bool = True) -> None:
    """
    Visualize image preprocessing pipeline
    """
    # Load original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized = cv2.resize(original, target_size)
    
    # Normalize (if requested)
    if normalize:
        normalized = resized.astype(np.float32) / 255.0
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
    else:
        normalized = resized
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title(f'Original\n{original.shape}')
    axes[0].axis('off')
    
    axes[1].imshow(resized)
    axes[1].set_title(f'Resized\n{resized.shape}')
    axes[1].axis('off')
    
    if normalize:
        # Show normalized image (clip values for display)
        display_normalized = np.clip(normalized * std + mean, 0, 1)
        axes[2].imshow(display_normalized)
        axes[2].set_title(f'Normalized\nMean: {normalized.mean():.3f}')
    else:
        axes[2].imshow(normalized)
        axes[2].set_title('No normalization')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== GENERATE TEST IMAGES ====================

def generate_test_images(output_dir: str, n_images: int = 50, 
                       metadata_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate synthetic test images with different patterns
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    image_info = []
    
    for i in range(n_images):
        # Generate random image characteristics
        width = np.random.choice([224, 256, 512])
        height = np.random.choice([224, 256, 512])
        
        # Create different patterns based on target
        if metadata_df is not None and i < len(metadata_df):
            target = metadata_df.iloc[i]['target']
        else:
            target = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Generate image based on target
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        if target == 1:
            # Positive class - more structured patterns
            pattern_type = np.random.choice(['circles', 'lines', 'gradient'])
            
            if pattern_type == 'circles':
                for _ in range(5):
                    center = (np.random.randint(0, width), np.random.randint(0, height))
                    radius = np.random.randint(20, 50)
                    color = tuple(np.random.randint(100, 255, 3).tolist())
                    cv2.circle(img, center, radius, color, -1)
            
            elif pattern_type == 'lines':
                for _ in range(10):
                    pt1 = (np.random.randint(0, width), np.random.randint(0, height))
                    pt2 = (np.random.randint(0, width), np.random.randint(0, height))
                    color = tuple(np.random.randint(100, 255, 3).tolist())
                    cv2.line(img, pt1, pt2, color, 3)
            
            else:  # gradient
                for y in range(height):
                    for x in range(width):
                        img[y, x] = [x * 255 // width, y * 255 // height, 128]
        
        else:
            # Negative class - more noise
            noise = np.random.randint(0, 100, (height, width, 3), dtype=np.uint8)
            img = noise
        
        # Save image
        filename = f'test_image_{i:04d}.jpg'
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Record info
        image_info.append({
            'image_id': i,
            'filename': filename,
            'path': str(filepath),
            'width': width,
            'height': height,
            'target': target,
            'pattern': pattern_type if target == 1 else 'noise'
        })
    
    return pd.DataFrame(image_info)

# ==================== IMAGE DATA EXPLORER CLASS ====================

class ImageDataExplorer:
    """
    Comprehensive image data exploration tool
    """
    def __init__(self, data_df: pd.DataFrame, image_col: str = 'path'):
        """
        Initialize explorer
        
        Parameters:
        - data_df: DataFrame with metadata and image paths
        - image_col: Column containing image paths
        """
        self.df = data_df.copy()
        self.image_col = image_col
        
    def explore_by_target(self, target_col: str, n_samples: int = 6):
        """
        Show sample images grouped by target value
        """
        target_values = self.df[target_col].unique()
        
        for target in sorted(target_values):
            subset = self.df[self.df[target_col] == target]
            sample = subset.sample(min(n_samples, len(subset)))
            
            print(f"\n{target_col} = {target} (n={len(subset)})")
            
            paths = sample[self.image_col].tolist()
            labels = [f"ID: {row['image_id']}" for _, row in sample.iterrows()]
            
            plot_image_grid(paths, labels=labels, n_cols=3, fig_width=12)
    
    def analyze_image_properties(self):
        """
        Analyze and visualize image properties
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Dimension scatter plot
        ax = axes[0, 0]
        ax.scatter(self.df['width'], self.df['height'], alpha=0.5)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_title('Image Dimensions')
        ax.grid(True)
        
        # Aspect ratio distribution
        ax = axes[0, 1]
        self.df['aspect_ratio'] = self.df['width'] / self.df['height']
        ax.hist(self.df['aspect_ratio'], bins=20, edgecolor='black')
        ax.set_xlabel('Aspect Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Aspect Ratio Distribution')
        
        # Size distribution
        ax = axes[1, 0]
        self.df['total_pixels'] = self.df['width'] * self.df['height']
        ax.hist(self.df['total_pixels'] / 1000000, bins=20, edgecolor='black')
        ax.set_xlabel('Megapixels')
        ax.set_ylabel('Count')
        ax.set_title('Image Size Distribution')
        
        # Target distribution by size
        ax = axes[1, 1]
        if 'target' in self.df.columns:
            for target in self.df['target'].unique():
                subset = self.df[self.df['target'] == target]
                ax.hist(subset['total_pixels'] / 1000000, alpha=0.5, 
                       label=f'Target={target}', bins=15)
            ax.set_xlabel('Megapixels')
            ax.set_ylabel('Count')
            ax.set_title('Size by Target')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_preprocessing_report(self, sample_size: int = 5):
        """
        Create a report showing preprocessing effects on sample images
        """
        sample = self.df.sample(min(sample_size, len(self.df)))
        
        for _, row in sample.iterrows():
            print(f"\nImage ID: {row.get('image_id', 'Unknown')}")
            print(f"Target: {row.get('target', 'Unknown')}")
            print(f"Original size: {row['width']} × {row['height']}")
            
            plot_image_preprocessing_pipeline(row[self.image_col])

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Photo Data Toolkit loaded successfully!")
    print("\nAvailable functions:")
    print("- load_image_info(folder)")
    print("- quick_image_summary(df)")
    print("- plot_image_grid(paths)")
    print("- ImageBrowser(df)")
    print("- visualize_image_with_data(path, metadata)")
    print("- analyze_image_stats(folder)")
    print("- generate_test_images(output_dir)")
    print("- ImageDataExplorer(df)")
