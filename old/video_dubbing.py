import moviepy.editor as mp
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
from math import ceil
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import srt
import datetime
import whisper
import torch
from tqdm import tqdm
import time
from TTS.api import TTS
import signal
import sys
import shutil
import tempfile
import gc
from concurrent.futures import ThreadPoolExecutor
import json
import psutil
import torch.cuda.memory
from time import sleep

class VideoDubber:
    def __init__(self, work_dir=None):
        self.TEMP_DIR = None
        self.whisper_proc = None
        self.voice_cloner = None
        self.progress = {}
        self.voice_samples = []
        self.sample_ratings = {}
        self.setup_work_directory(work_dir)
        
    def setup_work_directory(self, work_dir=None):
        """Initialize or use existing work directory"""
        if work_dir:
            self.TEMP_DIR = work_dir
        else:
            self.TEMP_DIR = tempfile.mkdtemp(prefix="video_dubbing_")
            
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "voice_references"), exist_ok=True)
        
        # Load progress if exists
        progress_file = os.path.join(self.TEMP_DIR, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                self.progress = json.load(f)
                print(f"\n📝 Found existing progress: {len(self.progress)} segments")
    
    def initialize_processors(self):
        """Initialize required processors"""
        print("\n🎯 Initializing processors...")
        self.whisper_proc = WhisperProcessor("medium")
        self.voice_cloner = VoiceCloner()
    
    def verify_voice_samples(self, reference_samples):
        """Interactive voice sample verification"""
        self.voice_samples = reference_samples
        self.sample_ratings = {}
        
        print("\n🎧 Voice Sample Verification")
        print("Rate each sample after listening (1-5 stars, 0 to reject)")
        
        while True:
            print("\nOptions:")
            print("p1-p5: Play sample 1-5")
            print("r1-r5: Rate sample 1-5")
            print("b: Play best rated sample")
            print("l: List current ratings")
            print("c: Continue with approved samples")
            print("n: Extract new samples")
            print("q: Quit")
            
            choice = input("\nYour choice: ").lower().strip()
            
            if choice == 'q':
                return False
                
            elif choice == 'c':
                good_samples = [s for s, r in self.sample_ratings.items() if r > 2]
                if not good_samples:
                    print("\n❌ No samples rated above 2 stars. Please rate samples or extract new ones.")
                    continue
                print(f"\n✓ Continuing with {len(good_samples)} approved samples")
                self.voice_samples = good_samples
                return True
                
            elif choice == 'n':
                print("\n🔄 Extracting new samples...")
                # Use different parameters for extraction
                new_samples = self.voice_cloner.extract_voice_references(
                    self.current_video_path,
                    min_duration=4,  # Try different duration
                    max_duration=8,
                    silence_thresh=-45  # Try different threshold
                )
                self.voice_samples = new_samples
                self.sample_ratings = {}
                print("\n✓ New samples extracted. Please rate them.")
                
            elif choice == 'l':
                print("\nCurrent ratings:")
                for i, sample in enumerate(self.voice_samples):
                    rating = self.sample_ratings.get(sample, "Not rated")
                    if rating != "Not rated":
                        rating = "⭐" * rating
                    print(f"Sample {i+1}: {rating}")
                
            elif choice == 'b':
                best_sample = max(self.sample_ratings.items(), key=lambda x: x[1])[0] if self.sample_ratings else None
                if best_sample:
                    print(f"\n▶️ Playing best rated sample...")
                    self.play_audio(best_sample)
                else:
                    print("\n❌ No rated samples yet")
                
            elif choice.startswith('p') and choice[1:].isdigit():
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(self.voice_samples):
                    print(f"\n▶️ Playing sample {idx+1}...")
                    self.play_audio(self.voice_samples[idx])
                else:
                    print("\n❌ Invalid sample number")
                    
            elif choice.startswith('r') and choice[1:].isdigit():
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(self.voice_samples):
                    try:
                        rating = int(input(f"Rate sample {idx+1} (0-5 stars): "))
                        if 0 <= rating <= 5:
                            self.sample_ratings[self.voice_samples[idx]] = rating
                            print(f"✓ Sample {idx+1} rated: {'⭐' * rating if rating > 0 else '❌'}")
                        else:
                            print("❌ Rating must be between 0 and 5")
                    except ValueError:
                        print("❌ Please enter a valid number")
                else:
                    print("\n❌ Invalid sample number")
            
            else:
                print("\n❌ Invalid choice. Please try again.")
    
    def process_segment(self, input_video, start_time, end_time):
        """Process a single video segment"""
        try:
            # Extract subtitles
            ru_subs, es_subs = self.whisper_proc.process_video_segment(
                input_video, start_time, end_time
            )
            
            # Generate audio
            temp_audio = os.path.join(self.TEMP_DIR, "audio", f"temp_audio_{start_time}_{end_time}.wav")
            if not self.voice_cloner.generate_timed_audio(es_subs, temp_audio, input_video):
                raise RuntimeError("Failed to generate audio")
            
            # Create video segment
            temp_output = os.path.join(self.TEMP_DIR, "video", f"temp_output_{start_time}.mp4")
            self.merge_audio_video(input_video, temp_audio, temp_output, start_time, end_time)
            
            return temp_output
            
        except Exception as e:
            print(f"\n⚠️ Error processing segment {start_time}-{end_time}: {str(e)}")
            return None
    
    def merge_audio_video(self, video_path, audio_path, output_path, start_time, end_time):
        """Merge audio and video segments"""
        video_segment = mp.VideoFileClip(video_path).subclip(start_time, end_time)
        audio = mp.AudioFileClip(audio_path)
        final_segment = video_segment.set_audio(audio)
        final_segment.write_videofile(output_path, logger=None)
        
        # Cleanup
        video_segment.close()
        audio.close()
        final_segment.close()
    
    def dub_video(self, input_video_path, output_video_path, segment_duration=300):
        """Main dubbing process"""
        try:
            self.current_video_path = input_video_path  # Store for re-extraction
            # Initialize
            self.initialize_processors()
            
            # Extract and verify voice samples
            reference_samples = self.voice_cloner.extract_voice_references(input_video_path)
            if not self.verify_voice_samples(reference_samples):
                print("\n⚠️ Process cancelled by user")
                return
            
            # Process video segments
            video = mp.VideoFileClip(input_video_path)
            duration = video.duration
            video.close()
            
            total_segments = ceil(duration/segment_duration)
            dubbed_segments = []
            
            with tqdm(total=total_segments, desc="Processing segments") as pbar:
                for start_time in range(0, int(duration), segment_duration):
                    end_time = min(start_time + segment_duration, duration)
                    segment_key = f"segment_{start_time}_{end_time}"
                    
                    # Check existing progress
                    if segment_key in self.progress:
                        dubbed_segments.append(self.progress[segment_key]["output"])
                        pbar.update(1)
                        continue
                    
                    # Process new segment
                    output = self.process_segment(input_video_path, start_time, end_time)
                    if output:
                        dubbed_segments.append(output)
                        self.progress[segment_key] = {
                            "output": output,
                            "completed": True
                        }
                        self.save_progress()
                    pbar.update(1)
            
            # Concatenate final video
            self.concatenate_segments(dubbed_segments, output_video_path)
            
        except Exception as e:
            print(f"\n❌ Error during processing: {str(e)}")
            raise

    def save_progress(self):
        """Save current progress to file"""
        progress_file = os.path.join(self.TEMP_DIR, "progress.json")
        with open(progress_file, 'w') as f:
            json.dump(self.progress, f)
    
    @staticmethod
    def play_audio(file_path):
        """Play audio file"""
        try:
            if sys.platform == "darwin":
                os.system(f"afplay {file_path}")
            elif sys.platform == "win32":
                os.system(f"start {file_path}")
            else:
                os.system(f"aplay {file_path}")
        except Exception as e:
            print(f"⚠️ Error playing audio: {e}")

if __name__ == "__main__":
    try:
        input_video = "russian_video.mp4"
        output_video = "spanish_dubbed_video.mp4"
        
        # Create dubber instance
        dubber = VideoDubber(work_dir=f"workdir_{os.path.splitext(os.path.basename(input_video))[0]}")
        
        # Start dubbing process
        dubber.dub_video(input_video, output_video, segment_duration=30)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")