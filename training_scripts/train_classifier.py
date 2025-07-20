#!/usr/bin/env python3
"""
Vertex AI Custom Training Script for Image Classification

This script is designed to run on Vertex AI for training image classification models.
It supports various model architectures and can be customized with different hyperparameters.
"""

import os
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import pandas as pd
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train image classification model on Vertex AI')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True,
                       help='GCS path to training CSV file')
    parser.add_argument('--validation_csv', type=str, required=True,
                       help='GCS path to validation CSV file')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'mobilenet', 'custom'],
                       help='Type of model architecture')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes (auto-detected if not specified)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    
    # Output arguments
    parser.add_argument('--model_dir', type=str, default='/tmp/model',
                       help='Directory to save the trained model')
    parser.add_argument('--export_dir', type=str, default='/tmp/export',
                       help='Directory to export the model for serving')
    
    return parser.parse_args()

def load_and_preprocess_data(csv_path: str, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data from CSV file.
    
    Args:
        csv_path: Path to CSV file with image paths and labels
        image_size: Target image size
        
    Returns:
        Tuple of (images, labels)
    """
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    
    for _, row in df.iterrows():
        try:
            # Load and preprocess image
            image = Image.open(row['image_path'])
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            image_array = np.array(image) / 255.0
            
            images.append(image_array)
            labels.append(row['class_id'])
            
        except Exception as e:
            logger.warning(f"Failed to load image {row['image_path']}: {e}")
            continue
    
    return np.array(images), np.array(labels)

def create_model(
    model_type: str,
    num_classes: int,
    image_size: int,
    dropout_rate: float
) -> keras.Model:
    """
    Create a classification model.
    
    Args:
        model_type: Type of model architecture
        num_classes: Number of classes
        image_size: Input image size
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=(image_size, image_size, 3))
    
    if model_type == 'efficientnet':
        # Use EfficientNetB0 as base
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        base_model.trainable = False  # Freeze base model initially
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
    elif model_type == 'resnet':
        # Use ResNet50 as base
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        base_model.trainable = False
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
    elif model_type == 'mobilenet':
        # Use MobileNetV2 as base
        base_model = keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        base_model.trainable = False
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
    else:  # custom
        # Simple CNN architecture
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks(model_dir: str) -> List[callbacks.Callback]:
    """Create training callbacks."""
    callbacks_list = [
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    return callbacks_list

def main():
    """Main training function."""
    global args
    args = parse_arguments()
    
    logger.info("Starting image classification training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.export_dir, exist_ok=True)
    
    # Load training data
    logger.info("Loading training data...")
    train_images, train_labels = load_and_preprocess_data(args.train_csv, args.image_size)
    
    logger.info("Loading validation data...")
    val_images, val_labels = load_and_preprocess_data(args.validation_csv, args.image_size)
    
    # Determine number of classes if not specified
    if args.num_classes is None:
        args.num_classes = len(np.unique(np.concatenate([train_labels, val_labels])))
    
    logger.info(f"Training data shape: {train_images.shape}")
    logger.info(f"Validation data shape: {val_images.shape}")
    logger.info(f"Number of classes: {args.num_classes}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        image_size=args.image_size,
        dropout_rate=args.dropout_rate
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks_list = create_callbacks(args.model_dir)
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_images,
        train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(args.model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    
    # Evaluate final model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(val_images, val_labels, verbose=0)
    logger.info(f"Final validation accuracy: {test_accuracy:.4f}")
    logger.info(f"Final validation loss: {test_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.h5')
    model.save(final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    # Export model for serving
    logger.info("Exporting model for serving...")
    export_path = os.path.join(args.export_dir, 'model')
    
    # Create serving signature
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, args.image_size, args.image_size, 3], dtype=tf.float32)])
    def serving_fn(inputs):
        return {'predictions': model(inputs)}
    
    # Save model with serving signature
    tf.saved_model.save(
        model,
        export_path,
        signatures={'serving_default': serving_fn}
    )
    
    logger.info(f"Model exported for serving to {export_path}")
    
    # Save model metadata
    metadata = {
        'model_type': args.model_type,
        'image_size': args.image_size,
        'num_classes': args.num_classes,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'final_accuracy': float(test_accuracy),
        'final_loss': float(test_loss),
        'training_completed_at': datetime.now().isoformat(),
        'export_path': export_path
    }
    
    metadata_path = os.path.join(args.model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main() 