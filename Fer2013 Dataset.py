import pandas as pd
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path settings
train_dir = r"C:\Users\ralfa\OneDrive\fer2013\train"
test_dir = r"C:\Users\ralfa\OneDrive\fer2013\test"
output_csv = "fer2013_dataset_enhanced.csv"
output_npy = "fer2013_arrays.npz"

# Emotion mapping (standard FER2013 mapping)
emotion_mapping = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

# Reverse emotion mapping for display
emotion_labels = {v: k for k, v in emotion_mapping.items()}


def process_single_image(args):
    """
    Process a single image (suitable for parallel processing)
    
    Args:
        args: Tuple containing (img_path, emotion_num, flip_correction, usage)
    
    Returns:
        Dictionary with processed image data or None if error occurs
    """
    img_path, emotion_num, flip_correction, usage = args
    
    try:
        # Open and convert image to grayscale
        img = Image.open(img_path).convert('L')
        
        # Apply flip correction if needed (for RTL/LTR issues)
        if flip_correction:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Resize to standard FER2013 size (48x48 pixels)
        img = img.resize((48, 48))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.uint8)
        
        # Convert to string format for CSV storage
        pixels_str = ' '.join(map(str, img_array.flatten()))
        
        return {
            'emotion': emotion_num,
            'pixels': pixels_str,
            'Usage': usage,
            'filename': os.path.basename(img_path)
        }
    
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None


def process_image_batch(image_batch, flip_correction=False, max_workers=4):
    """
    Process a batch of images using parallel processing
    
    Args:
        image_batch: List of image data tuples
        flip_correction: Whether to apply horizontal flip
        max_workers: Number of parallel workers
    
    Returns:
        List of processed image data
    """
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create list of futures for parallel execution
        futures = [
            executor.submit(process_single_image, (img_path, emotion_num, flip_correction, usage))
            for img_path, emotion_num, usage in image_batch
        ]
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results


def create_enhanced_dataset(flip_correction=False, max_workers=4, test_split=0.2):
    """
    Create enhanced dataset with parallel processing and better data splitting
    
    Args:
        flip_correction: Whether to apply horizontal flip correction
        max_workers: Number of parallel workers for image processing
        test_split: Ratio for test/validation split
    
    Returns:
        DataFrame containing the processed dataset
    """
    all_data = []
    image_batches = []
    
    logger.info("Collecting image data...")
    
    # Process training images
    for emotion_folder in os.listdir(train_dir):
        emotion_path = os.path.join(train_dir, emotion_folder)
        
        if os.path.isdir(emotion_path):
            emotion_num = emotion_mapping.get(emotion_folder.lower(), -1)
            
            if emotion_num == -1:
                logger.warning(f"Skipping unknown emotion folder: {emotion_folder}")
                continue
            
            # Get all image files in the emotion folder
            image_files = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            logger.info(f"Found {len(image_files)} images for {emotion_folder}")
            
            # Add to processing batch
            for img_file in image_files:
                img_path = os.path.join(emotion_path, img_file)
                image_batches.append((img_path, emotion_num, 'Training'))
    
    # Process test images
    for emotion_folder in os.listdir(test_dir):
        emotion_path = os.path.join(test_dir, emotion_folder)
        
        if os.path.isdir(emotion_path):
            emotion_num = emotion_mapping.get(emotion_folder.lower(), -1)
            
            if emotion_num == -1:
                logger.warning(f"Skipping unknown emotion folder: {emotion_folder}")
                continue
            
            image_files = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            logger.info(f"Found {len(image_files)} images for {emotion_folder}")
            
            for img_file in image_files:
                img_path = os.path.join(emotion_path, img_file)
                image_batches.append((img_path, emotion_num, 'PublicTest'))
    
    # Process batches using parallel processing
    logger.info("Starting image processing...")
    all_data = process_image_batch(image_batches, flip_correction, max_workers)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(all_data)
    
    # Split data into train/validation/test sets with stratification
    train_df, temp_df = train_test_split(
        df[df['Usage'] == 'Training'],
        test_size=test_split,
        random_state=42,
        stratify=df[df['Usage'] == 'Training']['emotion']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['emotion']
    )
    
    # Update usage columns
    train_df['Usage'] = 'Training'
    val_df['Usage'] = 'PrivateTest'
    test_df['Usage'] = 'PublicTest'
    
    # Combine all data including original test data
    final_df = pd.concat([
        train_df,
        val_df,
        test_df,
        df[df['Usage'] == 'PublicTest']
    ])
    
    # Shuffle the dataset for better training
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save as CSV
    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Save as numpy arrays for faster loading
    save_numpy_arrays(final_df)
    
    logger.info(f"✅ Dataset saved as: {output_csv}")
    logger.info(f"📊 Total images: {len(final_df)}")
    
    logger.info("📈 Class distribution:")
    # Log usage statistics
    for usage in final_df['Usage'].unique():
        usage_df = final_df[final_df['Usage'] == usage]
        logger.info(f"  {usage}: {len(usage_df)} images")
        logger.info(f"  Emotion distribution: {dict(usage_df['emotion'].value_counts().sort_index())}")
    
    return final_df


def save_numpy_arrays(df):
    """
    Save data as numpy arrays for faster loading during training
    
    Args:
        df: DataFrame containing the processed dataset
    """
    try:
        # Convert pixel strings to numpy arrays
        X = np.array([
            np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48)
            for pixels in tqdm(df['pixels'].values, desc="Converting to numpy arrays")
        ])
        
        y = df['emotion'].values
        usage = df['Usage'].values
        
        # Save as compressed numpy arrays
        np.savez_compressed(
            output_npy,
            X=X,
            y=y,
            usage=usage,
            emotion_labels=list(emotion_labels.values())
        )
        
        logger.info(f"✅ Numpy arrays saved as: {output_npy}")
    
    except Exception as e:
        logger.error(f"Error saving numpy arrays: {e}")


def load_numpy_data():
    """
    Load data from numpy file for faster access
    
    Returns:
        Tuple containing (X, y, usage, emotion_labels) or None if error occurs
    """
    try:
        # Set allow_pickle to True for loading emotion_labels array
        data = np.load(output_npy, allow_pickle=True)
        return data['X'], data['y'], data['usage'], data['emotion_labels']
    
    except Exception as e:
        logger.error(f"Error loading numpy data: {e}")
        return None, None, None, None


def verify_dataset(csv_path=None, npy_path=None, samples_per_class=3):
    """
    Verify the generated dataset by displaying sample images
    
    Args:
        csv_path: Path to CSV file (optional)
        npy_path: Path to numpy file (optional)
        samples_per_class: Number of samples to display per class
    """
    # Load data from numpy file if available, otherwise from CSV
    if npy_path and os.path.exists(npy_path):
        X, y, usage, loaded_emotion_labels = load_numpy_data()
        
        if X is not None:
            df = None
            # Convert loaded emotion labels to dictionary
            emotion_labels_dict = {i: label for i, label in enumerate(loaded_emotion_labels)}
        else:
            # Fall back to CSV if numpy loading fails
            logger.warning("Failed to load numpy data, falling back to CSV")
            df = pd.read_csv(csv_path)
            X = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48) for pixels in df['pixels'].values])
            y = df['emotion'].values
            emotion_labels_dict = emotion_labels
    
    else:
        # Load from CSV if no numpy path provided
        df = pd.read_csv(csv_path)
        X = np.array([np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48) for pixels in df['pixels'].values])
        y = df['emotion'].values
        emotion_labels_dict = emotion_labels
    
    # Check if we have valid data
    if X is None or len(X) == 0:
        logger.error("No valid data to visualize")
        return
    
    # Create subplot grid for sample images
    n_emotions = len(emotion_labels_dict)
    fig, axes = plt.subplots(n_emotions, samples_per_class, figsize=(samples_per_class * 2, n_emotions * 2))
    
    # Handle single row case
    if n_emotions == 1:
        axes = axes.reshape(1, -1)
    
    # Display samples for each emotion class
    for emotion_idx, emotion_name in emotion_labels_dict.items():
        emotion_indices = np.where(y == emotion_idx)[0]
        
        if len(emotion_indices) > 0:
            # Randomly select samples from this emotion class
            sample_indices = np.random.choice(
                emotion_indices,
                size=min(samples_per_class, len(emotion_indices)),
                replace=False
            )
            
            for col, idx in enumerate(sample_indices):
                if n_emotions > 1:
                    ax = axes[emotion_idx, col]
                else:
                    ax = axes[col]
                
                ax.imshow(X[idx], cmap='gray')
                ax.set_title(f"{emotion_name}\n({emotion_idx})")
                ax.axis('off')
        else:
            # Handle case where no samples exist for this emotion
            for col in range(samples_per_class):
                if n_emotions > 1:
                    ax = axes[emotion_idx, col]
                else:
                    ax = axes[col]
                
                ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print dataset statistics
    if df is not None:
        print("\n📊 Dataset Statistics:")
        print(f"Total samples: {len(df)}")
        print("\nUsage distribution:")
        print(df['Usage'].value_counts())
        print("\nEmotion distribution:")
        print(df['emotion'].value_counts().sort_index())
        
        # Save detailed report
        report = df.groupby(['Usage', 'emotion']).size().unstack(fill_value=0)
        report.to_csv('dataset_report.csv')
        print("✅ Dataset report saved as: dataset_report.csv")
    
    else:
        # Create a temporary dataframe for statistics if we loaded from numpy
        temp_df = pd.DataFrame({
            'emotion': y,
            'Usage': usage
        })
        
        print("\n📊 Dataset Statistics:")
        print(f"Total samples: {len(y)}")
        print("\nUsage distribution:")
        print(temp_df['Usage'].value_counts())
        print("\nEmotion distribution:")
        print(temp_df['emotion'].value_counts().sort_index())


def analyze_dataset_quality(df):
    """
    Analyze the quality of the generated dataset
    
    Args:
        df: DataFrame containing the dataset
    """
    print("🔍 Dataset Quality Analysis:")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    
    if missing_values.any():
        print(f"⚠️ Missing values found: {missing_values[missing_values > 0]}")
    else:
        print("✅ No missing values")
    
    # Check emotion distribution
    emotion_dist = df['emotion'].value_counts().sort_index()
    print(f"📈 Emotion distribution: {dict(emotion_dist)}")
    
    # Check data balance
    min_samples = emotion_dist.min()
    max_samples = emotion_dist.max()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    print(f"📊 Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 5:
        print("⚠️ Warning: Significant class imbalance detected")
        print("💡 Recommendation: Consider using class weights or data augmentation")
    elif imbalance_ratio > 2:
        print("ℹ️ Note: Moderate class imbalance present")
    else:
        print("✅ Well-balanced dataset")


# Main function
def main():
    """
    Main function to run the enhanced dataset creation pipeline
    """
    print("🚀 Starting enhanced dataset creation...")
    start_time = time.time()
    
    try:
        # Create the enhanced dataset
        df = create_enhanced_dataset(
            flip_correction=False,  # Set to True if RTL/LTR correction is needed
            max_workers=8,          # Number of parallel workers
            test_split=0.2          # Test/validation split ratio
        )
        
        # Analyze dataset quality
        analyze_dataset_quality(df)
        
        # Verify and visualize the dataset - handle potential errors
        try:
            verify_dataset(output_csv, output_npy)
        except Exception as e:
            logger.error(f"Error in verify_dataset: {e}")
            logger.info("Continuing with analysis...")
        
        end_time = time.time()
        print(f"⏰ Total execution time: {end_time - start_time:.2f} seconds")
        
        # Display imbalance warning
        emotion_dist = df['emotion'].value_counts().sort_index()
        min_samples = emotion_dist.min()
        max_samples = emotion_dist.max()
        
        if max_samples / min_samples > 5:
            print("\n🎯 IMPORTANT: Your dataset has significant class imbalance")
            print("   Consider using techniques like:")
            print("   - Class weighting in your model")
            print("   - Data augmentation for minority classes")
            print("   - Oversampling minority classes or undersampling majority classes")
    
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise


if __name__ == "__main__":
    main()