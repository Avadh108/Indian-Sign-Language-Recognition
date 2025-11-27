"""
Improved Training Pipeline with Augmentation Support
=====================================================
Train CNN-LSTM model with better regularization and augmented data
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import json
import os
from model import ISLWordClassifier
import matplotlib.pyplot as plt


class ImprovedISLTrainer:
    def __init__(self, data_path, model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def load_data(self):
        """Load preprocessed data"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)

        data = np.load(self.data_path, allow_pickle=True)

        X = data['X']
        y = data['y']
        word_to_idx = data['word_to_idx'].item()

        print(f"\n✓ Data loaded successfully!")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - Number of words: {len(word_to_idx)}")
        print(f"  - Words: {list(word_to_idx.keys())}")

        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n  Class distribution:")
        for word, idx in word_to_idx.items():
            count = counts[unique == idx][0] if idx in unique else 0
            print(f"    {word:15} : {count:3} samples")
        print()

        return X, y, word_to_idx

    def prepare_data(self, X, y, test_size=0.15, val_size=0.15):
        """Split data into train/val/test sets with stratification"""

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )

        print("="*70)
        print("DATA SPLIT")
        print("="*70)
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Val samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"\nTrain/Val/Test ratio: {X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.2f}/"
              f"{X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.2f}/"
              f"{X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0]):.2f}")
        print("="*70 + "\n")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(self, epochs=100, batch_size=8, use_class_weights=True):
        """Train the model with improved settings"""

        # Load data
        X, y, word_to_idx = self.load_data()

        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(X, y)

        # Build model
        num_classes = len(word_to_idx)
        sequence_length = X.shape[1]
        num_features = X.shape[2]

        print("="*70)
        print("BUILDING MODEL")
        print("="*70)

        model_builder = ISLWordClassifier(
            num_classes=num_classes,
            sequence_length=sequence_length,
            num_features=num_features
        )

        model = model_builder.build_model()
        model_builder.compile_model(learning_rate=0.0005)  # Lower learning rate
        model_builder.summary()

        # Calculate class weights for imbalanced data
        class_weights = None
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = dict(enumerate(class_weights))
            print("\n✓ Using class weights to handle imbalance")

        # Get callbacks with improved settings
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,  # Increased patience
                min_lr=1e-7,
                verbose=1
            ),
            # Add CSV logger
            keras.callbacks.CSVLogger(
                os.path.join(self.model_dir, 'training_log.csv')
            )
        ]

        # Train model
        print("\n" + "="*70)
        print(f"TRAINING FOR UP TO {epochs} EPOCHS")
        print("="*70)
        print(f"Batch size: {batch_size}")
        print(f"Initial learning rate: 0.0005")
        print("="*70 + "\n")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Evaluate on test set
        print("\n" + "="*70)
        print("FINAL EVALUATION ON TEST SET")
        print("="*70)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f"\n✓ Test Accuracy: {test_acc * 100:.2f}%")
        print(f"✓ Test Loss: {test_loss:.4f}")

        # Detailed predictions analysis
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

        from sklearn.metrics import classification_report, confusion_matrix

        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)

        idx_to_word = {v: k for k, v in word_to_idx.items()}
        target_names = [idx_to_word[i] for i in range(num_classes)]

        print(classification_report(y_test, y_pred, target_names=target_names))

        print("\n" + "="*70)
        print("CONFUSION MATRIX")
        print("="*70)
        cm = confusion_matrix(y_test, y_pred)
        print("\nRows=Actual, Columns=Predicted")
        print("Words:", target_names)
        print(cm)
        print("="*70 + "\n")

        # Save word mapping
        mapping_file = os.path.join(self.model_dir, 'word_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(word_to_idx, f, indent=2)

        print(f"✓ Model saved to: {self.model_dir}/best_model.h5")
        print(f"✓ Word mapping saved to: {mapping_file}")
        print(f"✓ Training log saved to: {self.model_dir}/training_log.csv")

        # Plot training history
        self.plot_history(history)

        print("\nReady for inference! Use the test_inference.py script.\n")

        return model, history, word_to_idx

    def plot_history(self, history):
        """Plot training history"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot accuracy
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)

            # Plot loss
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Val Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
            print(f"✓ Training plot saved to: {self.model_dir}/training_history.png")
            plt.close()
        except Exception as e:
            print(f"⚠ Could not save training plot: {e}")


def main():
    """Main function"""

    # Check if augmented data exists
    augmented_path = 'processed_data/isl_landmarks_augmented.npz'
    original_path = 'processed_data/isl_landmarks.npz'

    if os.path.exists(augmented_path):
        print("\n" + "="*70)
        print("FOUND AUGMENTED DATA - USING IT FOR TRAINING")
        print("="*70)
        data_path = augmented_path
    else:
        print("\n" + "="*70)
        print("NO AUGMENTED DATA FOUND - USING ORIGINAL DATA")
        print("="*70)
        print("\n⚠ RECOMMENDATION: Run data augmentation first:")
        print("  python 02b_augment_data.py")
        print("\nContinuing with original data...")
        data_path = original_path

    trainer = ImprovedISLTrainer(
        data_path=data_path,
        model_dir='models'
    )

    model, history, word_mapping = trainer.train(
        epochs=100,  # Increased epochs
        batch_size=8,  # Increased batch size for better gradients
        use_class_weights=True  # Handle class imbalance
    )


if __name__ == "__main__":
    main()
