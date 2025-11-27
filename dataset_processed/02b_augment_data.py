"""
Data Augmentation for ISL Word Recognition
===========================================
Improves model accuracy by creating synthetic variations of existing samples
"""

import numpy as np
import cv2
from pathlib import Path
import random


class ISLDataAugmenter:
    """
    Augment ISL hand landmark sequences to increase dataset diversity

    Augmentation techniques:
    1. Time-domain augmentation (speed, reverse, cropping)
    2. Spatial augmentation (rotation, scaling, translation, noise)
    3. Landmark-specific augmentation (jitter, dropout)
    """

    def __init__(self, augmentation_factor=5):
        """
        Args:
            augmentation_factor: How many augmented versions per original sample
        """
        self.augmentation_factor = augmentation_factor

    def augment_sequence(self, sequence):
        """
        Apply random augmentations to a landmark sequence

        Args:
            sequence: numpy array of shape (num_frames, 126)

        Returns:
            List of augmented sequences
        """
        augmented_sequences = []

        for _ in range(self.augmentation_factor):
            aug_seq = sequence.copy()

            # Apply random combination of augmentations
            if random.random() > 0.5:
                aug_seq = self.temporal_speed_variation(aug_seq)

            if random.random() > 0.5:
                aug_seq = self.spatial_rotation(aug_seq)

            if random.random() > 0.5:
                aug_seq = self.spatial_scaling(aug_seq)

            if random.random() > 0.6:
                aug_seq = self.add_gaussian_noise(aug_seq)

            if random.random() > 0.7:
                aug_seq = self.landmark_jitter(aug_seq)

            if random.random() > 0.8:
                aug_seq = self.temporal_crop_and_pad(aug_seq)

            augmented_sequences.append(aug_seq)

        return augmented_sequences

    def temporal_speed_variation(self, sequence, speed_range=(0.8, 1.2)):
        """
        Change signing speed by resampling frames
        """
        current_length = len(sequence)

        # Random speed factor
        speed = random.uniform(*speed_range)
        new_length = int(current_length * speed)

        if new_length < 10:
            new_length = 10

        # Resample using linear interpolation
        indices = np.linspace(0, current_length - 1, new_length)

        augmented = []
        for idx in indices:
            # Linear interpolation between frames
            low_idx = int(np.floor(idx))
            high_idx = min(int(np.ceil(idx)), current_length - 1)
            weight = idx - low_idx

            frame = (1 - weight) * sequence[low_idx] + weight * sequence[high_idx]
            augmented.append(frame)

        # Pad or crop back to original length
        augmented = np.array(augmented)
        return self._pad_or_crop(augmented, current_length)

    def spatial_rotation(self, sequence, angle_range=(-15, 15)):
        """
        Rotate hand landmarks around wrist
        """
        augmented = sequence.copy()
        angle = random.uniform(*angle_range) * np.pi / 180  # Convert to radians

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for frame_idx in range(len(augmented)):
            landmarks = augmented[frame_idx].reshape(-1, 3)

            # Rotate each hand separately
            for hand_idx in range(2):
                start_idx = hand_idx * 21
                end_idx = start_idx + 21
                hand_landmarks = landmarks[start_idx:end_idx]

                if np.any(hand_landmarks):
                    # Get wrist position as rotation center
                    wrist = hand_landmarks[0].copy()

                    # Translate to origin
                    hand_landmarks -= wrist

                    # Apply 2D rotation (x, y plane)
                    x = hand_landmarks[:, 0]
                    y = hand_landmarks[:, 1]

                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a

                    hand_landmarks[:, 0] = new_x
                    hand_landmarks[:, 1] = new_y

                    # Translate back
                    hand_landmarks += wrist

                    landmarks[start_idx:end_idx] = hand_landmarks

            augmented[frame_idx] = landmarks.flatten()

        return augmented

    def spatial_scaling(self, sequence, scale_range=(0.9, 1.1)):
        """
        Scale hand size
        """
        augmented = sequence.copy()
        scale = random.uniform(*scale_range)

        for frame_idx in range(len(augmented)):
            landmarks = augmented[frame_idx].reshape(-1, 3)

            for hand_idx in range(2):
                start_idx = hand_idx * 21
                end_idx = start_idx + 21
                hand_landmarks = landmarks[start_idx:end_idx]

                if np.any(hand_landmarks):
                    wrist = hand_landmarks[0].copy()
                    hand_landmarks -= wrist
                    hand_landmarks *= scale
                    hand_landmarks += wrist
                    landmarks[start_idx:end_idx] = hand_landmarks

            augmented[frame_idx] = landmarks.flatten()

        return augmented

    def add_gaussian_noise(self, sequence, noise_std=0.01):
        """
        Add small Gaussian noise to landmarks
        """
        noise = np.random.normal(0, noise_std, sequence.shape)
        return sequence + noise

    def landmark_jitter(self, sequence, jitter_std=0.02):
        """
        Add random jitter to individual landmarks
        """
        augmented = sequence.copy()

        for frame_idx in range(len(augmented)):
            landmarks = augmented[frame_idx].reshape(-1, 3)

            # Random jitter for each landmark
            jitter = np.random.normal(0, jitter_std, landmarks.shape)
            landmarks += jitter

            augmented[frame_idx] = landmarks.flatten()

        return augmented

    def temporal_crop_and_pad(self, sequence):
        """
        Randomly crop and pad sequence
        """
        current_length = len(sequence)

        # Random crop 80-100% of sequence
        crop_ratio = random.uniform(0.8, 1.0)
        crop_length = int(current_length * crop_ratio)

        # Random start position
        max_start = current_length - crop_length
        start_idx = random.randint(0, max(0, max_start))

        cropped = sequence[start_idx:start_idx + crop_length]

        # Pad back to original length
        return self._pad_or_crop(cropped, current_length)

    def _pad_or_crop(self, sequence, target_length):
        """Helper to pad or crop sequence to target length"""
        current_length = len(sequence)

        if current_length == target_length:
            return sequence
        elif current_length > target_length:
            # Crop by sampling evenly
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return sequence[indices]
        else:
            # Pad with last frame
            padding = np.repeat([sequence[-1]], target_length - current_length, axis=0)
            return np.vstack([sequence, padding])

    def reverse_sequence(self, sequence):
        """
        Reverse temporal order (for some signs this is valid)
        """
        return sequence[::-1].copy()


def augment_dataset(input_npz, output_npz, augmentation_factor=5):
    """
    Augment entire dataset and save augmented version

    Args:
        input_npz: Path to original preprocessed data
        output_npz: Path to save augmented data
        augmentation_factor: How many augmented versions per sample
    """
    print("\n" + "="*70)
    print("DATA AUGMENTATION")
    print("="*70)

    # Load original data
    data = np.load(input_npz, allow_pickle=True)
    X_original = data['X']
    y_original = data['y']
    word_to_idx = data['word_to_idx'].item()

    print(f"\nOriginal dataset:")
    print(f"  - Samples: {len(X_original)}")
    print(f"  - Words: {len(word_to_idx)}")
    print(f"  - Shape: {X_original.shape}")

    # Initialize augmenter
    augmenter = ISLDataAugmenter(augmentation_factor=augmentation_factor)

    # Store augmented data
    X_augmented = []
    y_augmented = []

    # Augment each sample
    print(f"\nAugmenting with factor {augmentation_factor}...")

    for idx, (sequence, label) in enumerate(zip(X_original, y_original)):
        # Keep original
        X_augmented.append(sequence)
        y_augmented.append(label)

        # Generate augmented versions
        augmented_sequences = augmenter.augment_sequence(sequence)

        for aug_seq in augmented_sequences:
            X_augmented.append(aug_seq)
            y_augmented.append(label)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(X_original)} samples...")

    # Convert to numpy arrays
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)

    print(f"\n✓ Augmentation complete!")
    print(f"  - Original samples: {len(X_original)}")
    print(f"  - Augmented samples: {len(X_augmented)}")
    print(f"  - Expansion: {len(X_augmented) / len(X_original):.1f}x")

    # Save augmented dataset
    np.savez_compressed(
        output_npz,
        X=X_augmented,
        y=y_augmented,
        word_to_idx=word_to_idx,
        target_length=X_original.shape[1]
    )

    print(f"\n✓ Saved augmented dataset to: {output_npz}")
    print("="*70 + "\n")

    return X_augmented, y_augmented


if __name__ == "__main__":
    # Augment your dataset
    augment_dataset(
        input_npz='processed_data/isl_landmarks.npz',
        output_npz='processed_data/isl_landmarks_augmented.npz',
        augmentation_factor=5  # 5x more data
    )

    print("Next step: Train model with augmented data!")
    print("  python 04_train_model.py")
