"""
Improved CNN-LSTM Model Architecture
=====================================
Enhanced model with better regularization for small datasets
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import os


class ImprovedISLWordClassifier:
    def __init__(self, num_classes=10, sequence_length=30, num_features=126):
        """
        Improved CNN-LSTM model with better regularization

        Args:
            num_classes: Number of word classes
            sequence_length: Sequence length
            num_features: Feature dimension (126 for 2 hands × 21 landmarks × 3 coords)
        """
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None

    def build_model(self):
        """Build improved CNN-LSTM architecture"""

        inputs = keras.Input(shape=(self.sequence_length, self.num_features))

        # Reshape for 1D CNN
        x = layers.Reshape((self.sequence_length, self.num_features, 1))(inputs)

        # 1D CNN for spatial feature extraction (WITH REGULARIZATION)
        x = layers.TimeDistributed(
            layers.Conv1D(
                32, kernel_size=3, activation='relu', padding='same',
                kernel_regularizer=regularizers.l2(0.001)
            )
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
        x = layers.TimeDistributed(layers.Dropout(0.3))(x)

        x = layers.TimeDistributed(
            layers.Conv1D(
                64, kernel_size=3, activation='relu', padding='same',
                kernel_regularizer=regularizers.l2(0.001)
            )
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
        x = layers.TimeDistributed(layers.Dropout(0.4))(x)

        x = layers.TimeDistributed(layers.GlobalMaxPooling1D())(x)

        # Bidirectional LSTM for temporal modeling (WITH REGULARIZATION)
        x = layers.Bidirectional(
            layers.LSTM(
                64, return_sequences=True, 
                dropout=0.4, recurrent_dropout=0.3,
                kernel_regularizer=regularizers.l2(0.001)
            )
        )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Bidirectional(
            layers.LSTM(
                32, return_sequences=False,
                dropout=0.4, recurrent_dropout=0.3,
                kernel_regularizer=regularizers.l2(0.001)
            )
        )(x)
        x = layers.BatchNormalization()(x)

        # Dense layers (WITH REGULARIZATION)
        x = layers.Dense(
            64, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(
            32, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        )(x)
        x = layers.Dropout(0.4)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model

    def compile_model(self, learning_rate=0.0005):
        """Compile the model with lower learning rate"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_callbacks(self, model_dir='models'):
        """Get training callbacks (kept for compatibility)"""
        os.makedirs(model_dir, exist_ok=True)

        return [
            keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]

    def summary(self):
        """Print model summary"""
        if self.model:
            print("\n" + "="*70)
            print("IMPROVED MODEL ARCHITECTURE")
            print("="*70 + "\n")
            self.model.summary()
            print("\n" + "="*70)
            print("KEY IMPROVEMENTS:")
            print("  ✓ L2 regularization on all layers")
            print("  ✓ Increased dropout (0.3-0.5)")
            print("  ✓ Recurrent dropout in LSTMs")
            print("  ✓ Batch normalization throughout")
            print("  ✓ Smaller model capacity (prevents overfitting)")
            print("="*70)
        else:
            print("Model not built yet. Call build_model() first.")


def main():
    """Main function"""
    # Build model
    model_builder = ImprovedISLWordClassifier(
        num_classes=10,  # Updated for 10 words
        sequence_length=30,
        num_features=126
    )

    model = model_builder.build_model()
    model_builder.compile_model()
    model_builder.summary()

    # Print parameter count
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    main()
