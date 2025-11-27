"""
Preprocessing Pipeline - Extract Landmarks
===========================================
Extract hand landmarks from collected videos for 5-word prototype
"""

import mediapipe as mp
import cv2
import numpy as np
import os
import json
from pathlib import Path
import pickle

class LandmarkExtractor:
    def __init__(self):
        """Initialize MediaPipe hand detector"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_frames(self, frames_dir):
        """
        Extract landmarks from a sequence of frames

        Returns:
            sequence: numpy array of shape (num_frames, 126)
        """
        frames = sorted(list(Path(frames_dir).glob('frame_*.jpg')))
        landmark_sequence = []

        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            # Extract landmarks
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Pad if only one hand
                while len(landmarks) < 126:
                    landmarks.append(0.0)

                landmark_sequence.append(np.array(landmarks[:126]))
            else:
                # No hands detected - use zeros
                landmark_sequence.append(np.zeros(126))

        return np.array(landmark_sequence) if landmark_sequence else None

    def normalize_sequence(self, sequence):
        """Normalize landmark sequence"""
        if sequence is None or len(sequence) == 0:
            return None

        normalized_sequence = []

        for frame_landmarks in sequence:
            landmarks = frame_landmarks.reshape(-1, 3)  # (42, 3)

            # Normalize each hand separately
            for hand_idx in range(2):
                start_idx = hand_idx * 21
                end_idx = start_idx + 21
                hand_landmarks = landmarks[start_idx:end_idx].copy()

                if np.any(hand_landmarks):
                    wrist = hand_landmarks[0]
                    hand_landmarks -= wrist

                    distances = np.linalg.norm(hand_landmarks, axis=1)
                    max_distance = np.max(distances)
                    if max_distance > 0:
                        hand_landmarks /= max_distance

                    landmarks[start_idx:end_idx] = hand_landmarks

            normalized_sequence.append(landmarks.flatten())

        return np.array(normalized_sequence)

    def pad_sequence(self, sequence, target_length=30):
        """Pad or truncate sequence to target length"""
        if sequence is None:
            return None

        current_length = len(sequence)

        if current_length == target_length:
            return sequence
        elif current_length > target_length:
            # Sample evenly
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return sequence[indices]
        else:
            # Pad with last frame
            padding = np.repeat([sequence[-1]], target_length - current_length, axis=0)
            return np.vstack([sequence, padding])

    def process_dataset(self, data_dir, output_file, target_length=30):
        """Process entire dataset"""
        print("\n" + "="*70)
        print("PROCESSING DATASET - EXTRACTING LANDMARKS")
        print("="*70)

        data = []
        labels = []
        word_to_idx = {}
        idx_counter = 0

        word_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]

        print(f"\nFound {len(word_dirs)} words\n")

        for word_dir in sorted(word_dirs):
            word = word_dir.name

            if word not in word_to_idx:
                word_to_idx[word] = idx_counter
                idx_counter += 1

            print(f"Processing word: '{word}'", end=" | ", flush=True)

            # Process each sample
            sample_dirs = sorted([d for d in word_dir.iterdir() if d.is_dir()])
            sample_count = 0

            for sample_dir in sample_dirs:
                # Extract landmarks
                sequence = self.extract_from_frames(sample_dir)

                if sequence is None or len(sequence) == 0:
                    continue

                # Normalize
                normalized = self.normalize_sequence(sequence)

                if normalized is None:
                    continue

                # Pad/truncate
                padded = self.pad_sequence(normalized, target_length)

                data.append(padded)
                labels.append(word_to_idx[word])
                sample_count += 1

            print(f"Processed {sample_count} samples")

        # Convert to numpy arrays
        X = np.array(data)
        y = np.array(labels)

        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save data
        np.savez_compressed(output_file, X=X, y=y, word_to_idx=word_to_idx, target_length=target_length)

        print(f"\n{'='*70}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Total samples: {len(X)}")
        print(f"✓ Total words: {len(word_to_idx)}")
        print(f"✓ Sequence length: {target_length}")
        print(f"✓ Feature shape: {X.shape}")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*70}\n")

        # Save word mapping
        mapping_file = output_file.replace('.npz', '_word_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(word_to_idx, f, indent=2)
        print(f"✓ Word mapping saved to: {mapping_file}\n")

        return X, y, word_to_idx


def main():
    """Main function"""
    extractor = LandmarkExtractor()

    # Process dataset
    X, y, word_mapping = extractor.process_dataset(
        data_dir='isl_dataset',
        output_file='processed_data/isl_landmarks.npz',
        target_length=30  # 30 frames for prototype
    )

    print("\nDataset ready for training!")
    print(f"Words: {list(word_mapping.keys())}")


if __name__ == "__main__":
    main()
