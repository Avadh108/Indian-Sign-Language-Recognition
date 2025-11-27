
"""
ISL Word-Level Data Collection System
======================================
Complete script for collecting Indian Sign Language word videos via webcam.

Features:
- Automated countdown before recording
- Progress bar during recording
- Real-time hand detection visualization
- Organized file structure
- Metadata tracking
- Easy-to-use controls
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime
import time

class ISLDataCollector:
    def __init__(self, data_dir='isl_dataset', fps=30):
        """
        Initialize the ISL data collector

        Args:
            data_dir: Directory to save collected data
            fps: Frames per second for video capture
        """
        self.data_dir = data_dir
        self.fps = fps

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create main directory
        os.makedirs(data_dir, exist_ok=True)

        # Initialize metadata
        self.metadata = {
            "collection_date": datetime.now().isoformat(),
            "fps": fps,
            "words": {}
        }

        print("\n" + "="*70)
        print("ISL WORD-LEVEL DATA COLLECTION SYSTEM")
        print("="*70)

    def collect_word(self, word_name, num_samples=10, frames_per_sample=30, 
                     countdown_seconds=3):
        """
        Collect video samples for a specific word

        Args:
            word_name: Name of the ISL word to collect
            num_samples: Number of video samples to collect
            frames_per_sample: Number of frames per video (45 = 1.5 sec @ 30fps)
            countdown_seconds: Countdown time before recording starts
        """
        # Create word directory
        word_dir = os.path.join(self.data_dir, word_name)
        os.makedirs(word_dir, exist_ok=True)

        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Collection state variables
        sample_count = 0
        recording = False
        countdown = 0
        countdown_active = False
        frame_buffer = []
        last_save_time = 0

        # Display instructions
        print(f"\n{'='*70}")
        print(f"COLLECTING WORD: '{word_name}'")
        print(f"{'='*70}")
        print(f"\nTarget: {num_samples} samples ({frames_per_sample} frames each)")
        print("\nCONTROLS:")
        print("  - Press 'R' to start recording (with countdown)")
        print("  - Press 'S' to skip current sample")
        print("  - Press 'N' to finish and move to next word")
        print("  - Press 'Q' to quit completely")
        print("\nTIPS:")
        print("  - Keep your hands visible in the camera")
        print("  - Perform the sign naturally and consistently")
        print("  - Wait for the green 'RECORDING' indicator")
        print("\n" + "-"*70 + "\n")

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from webcam")
                break

            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands
            results = self.hands.process(rgb_frame)

            # Draw hand landmarks with better styling
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Countdown logic
            if countdown_active:
                seconds_left = countdown // self.fps

                # Large countdown number
                cv2.putText(
                    frame, 
                    str(seconds_left + 1), 
                    (frame.shape[1]//2 - 50, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    5, 
                    (0, 165, 255),  # Orange
                    10
                )

                # Instruction text
                cv2.putText(
                    frame,
                    "GET READY!",
                    (frame.shape[1]//2 - 150, frame.shape[0]//2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 165, 255),
                    3
                )

                countdown -= 1

                if countdown <= 0:
                    countdown_active = False
                    recording = True
                    frame_buffer = []
                    print(f"  🎥 Recording sample {sample_count + 1}...", end="", flush=True)

            # Recording logic
            elif recording:
                # Only record frames where hand is detected
                if hand_detected:
                    frame_buffer.append(frame.copy())

                # Progress calculation
                progress = len(frame_buffer) / frames_per_sample

                # Top banner - RECORDING indicator
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    "● RECORDING",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),  # Green
                    3
                )

                # Progress bar
                bar_width = 600
                bar_height = 30
                bar_x = frame.shape[1] - bar_width - 20
                bar_y = 25

                # Background
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (80, 80, 80),
                    -1
                )

                # Progress fill
                fill_width = int(bar_width * progress)
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + fill_width, bar_y + bar_height),
                    (0, 255, 0),
                    -1
                )

                # Progress text
                progress_text = f"{len(frame_buffer)}/{frames_per_sample}"
                cv2.putText(
                    frame,
                    progress_text,
                    (bar_x + bar_width//2 - 50, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Save when enough frames collected
                if len(frame_buffer) >= frames_per_sample:
                    self._save_video_sequence(word_name, sample_count, frame_buffer)
                    sample_count += 1
                    recording = False
                    frame_buffer = []
                    last_save_time = time.time()
                    print(f" ✓ Saved! ({sample_count}/{num_samples})")
                    time.sleep(0.3)  # Brief pause

            # Idle state - waiting for user
            else:
                current_time = time.time()

                # Top info banner
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (40, 40, 40), -1)

                # Word name
                cv2.putText(
                    frame,
                    f"Word: {word_name}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                # Progress
                cv2.putText(
                    frame,
                    f"Progress: {sample_count}/{num_samples}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

                # Hand detection indicator
                status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
                status_text = "Hand Detected ✓" if hand_detected else "No Hand Detected ✗"
                cv2.putText(
                    frame,
                    status_text,
                    (frame.shape[1] - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2
                )

                # Instruction box
                if current_time - last_save_time > 1.5:  # Show after brief pause
                    instruction = "Press 'R' to Start Recording"
                    text_size = cv2.getTextSize(
                        instruction, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, 
                        3
                    )[0]

                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] - 80

                    # Background box
                    cv2.rectangle(
                        frame,
                        (text_x - 20, text_y - 40),
                        (text_x + text_size[0] + 20, text_y + 10),
                        (0, 100, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        instruction,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (255, 255, 255),
                        3
                    )

            # Display frame
            cv2.imshow('ISL Data Collection', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') and not recording and not countdown_active:
                # Start countdown
                countdown = countdown_seconds * self.fps
                countdown_active = True

            elif key == ord('s') and recording:
                # Skip current sample
                recording = False
                frame_buffer = []
                print("  ⊗ Sample skipped")

            elif key == ord('n'):
                # Move to next word
                print(f"\n  → Moving to next word. Collected {sample_count} samples.")
                break

            elif key == ord('q'):
                # Quit completely
                print("\n  ✗ Collection stopped by user.")
                cap.release()
                cv2.destroyAllWindows()
                return sample_count

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Update metadata
        self.metadata["words"][word_name] = {
            "samples_collected": sample_count,
            "frames_per_sample": frames_per_sample,
            "collection_date": datetime.now().isoformat()
        }
        self._save_metadata()

        print(f"\n{'='*70}")
        print(f"✓ COMPLETED: '{word_name}' - {sample_count} samples collected")
        print(f"{'='*70}\n")

        return sample_count

    def _save_video_sequence(self, word, sample_idx, frames):
        """Save video sequence as individual frames"""
        sample_dir = os.path.join(
            self.data_dir,
            word,
            f"{word}_sample_{sample_idx:04d}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        # Save frames
        for frame_idx, frame in enumerate(frames):
            frame_path = os.path.join(sample_dir, f"frame_{frame_idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)

        # Save sample metadata
        sample_meta = {
            "word": word,
            "sample_index": sample_idx,
            "num_frames": len(frames),
            "timestamp": datetime.now().isoformat()
        }

        meta_path = os.path.join(sample_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(sample_meta, f, indent=2)

    def _save_metadata(self):
        """Save collection metadata"""
        meta_path = os.path.join(self.data_dir, "collection_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def collect_multiple_words(self, word_list, samples_per_word=10):
        """
        Collect multiple words in sequence

        Args:
            word_list: List of word names to collect
            samples_per_word: Number of samples per word
        """
        print(f"\n{'='*70}")
        print(f"MULTI-WORD COLLECTION")
        print(f"{'='*70}")
        print(f"Total words: {len(word_list)}")
        print(f"Samples per word: {samples_per_word}")
        print(f"Total samples to collect: {len(word_list) * samples_per_word}")
        print(f"{'='*70}\n")

        for idx, word in enumerate(word_list):
            print(f"\n[{idx + 1}/{len(word_list)}] Starting collection for: '{word}'")
            self.collect_word(word, num_samples=samples_per_word)

            # Prompt to continue
            if idx < len(word_list) - 1:
                print(f"\nNext word: '{word_list[idx + 1]}'")
                input("Press ENTER to continue, or Ctrl+C to stop...")

        print(f"\n{'='*70}")
        print("ALL WORDS COLLECTED!")
        print(f"{'='*70}")
        print(f"Dataset saved in: {self.data_dir}")
        print(f"Total words: {len(self.metadata['words'])}")


def main():
    """Main function to run the data collection"""

    # Initialize collector
    collector = ISLDataCollector(
        data_dir='isl_dataset',  # Change this to your preferred directory
        fps=30
    )

    # Define your word list
    # OPTION 1: Collect a single word
    # collector.collect_word('hello', num_samples=50)

    # OPTION 2: Collect multiple words
    word_list = [
        # Greetings & Common
        'hello', 'thank_you', 'sorry',
        'namaste',

        # Family
        # 'family',

        # Basic Actions
        # 'eat', 'drink',

        # Emotions
        'love',

        # Questions
        # 'what', 'when', 'who'
    ]

    # Collect all words
    collector.collect_multiple_words(word_list, samples_per_word=10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCollection stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
