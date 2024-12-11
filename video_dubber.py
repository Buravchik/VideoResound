import os
import sys
import json
import time
import signal
import tempfile
from math import ceil
from tqdm import tqdm
import moviepy.editor as mp
import re

from processors.whisper_processor import WhisperProcessor
from processors.voice_cloner import VoiceCloner
from utils.audio import verify_voice_samples, play_audio

class VideoDubber:
    def __init__(self, work_dir=None):
        self.TEMP_DIR = None
        self.whisper_proc = None
        self.voice_cloner = None
        self.progress = {}
        self.progress_file = None
        self.reference_audio = None
        if work_dir:
            self.setup_work_directory(work_dir)
            self.initialize_processors()  # Initialize processors right after work_dir setup
    
    def setup_work_directory(self, work_dir=None):
        """Initialize or use existing work directory"""
        if work_dir:
            self.TEMP_DIR = work_dir
        else:
            self.TEMP_DIR = tempfile.mkdtemp(prefix="video_dubbing_")
            
        # Create directory structure
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "voice_references/extracted"), exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "voice_references/approved"), exist_ok=True)
        os.makedirs(os.path.join(self.TEMP_DIR, "subtitles"), exist_ok=True)  # Add subtitles directory
        
        # Load progress if exists
        self.progress_file = os.path.join(self.TEMP_DIR, "progress.json")
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
                print(f"\n📝 Found existing progress: {len(self.progress)} segments")
        else:
            self.progress = {}
        
        # Diagnostic check
        print("\n📁 Work directory structure:")
        for root, dirs, files in os.walk(self.TEMP_DIR):
            level = root.replace(self.TEMP_DIR, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")
    
    def initialize_processors(self):
        """Initialize processing components"""
        print("\n🔧 Initializing processors...")
        try:
            if not self.whisper_proc:
                self.whisper_proc = WhisperProcessor()
                self.whisper_proc.load_translation_cache(self.TEMP_DIR)
                print("✓ Whisper processor initialized")
                
            if not self.voice_cloner:
                self.voice_cloner = VoiceCloner()
                self.voice_cloner.set_work_directory(self.TEMP_DIR)
                print("✓ Voice cloner initialized")
        except Exception as e:
            print(f"❌ Failed to initialize processors: {str(e)}")
            raise
    
    def process_segment(self, input_video, start_time, end_time):
        """Process a single video segment"""
        try:
            # Extract subtitles
            ru_subs, es_subs = self.whisper_proc.process_video_segment(
                input_video, 
                start_time, 
                end_time,
                self.TEMP_DIR
            )
            
            # Generate audio using the stored reference audio path
            temp_audio = os.path.join(self.TEMP_DIR, "audio", f"temp_audio_{start_time}_{end_time}.wav")
            if not self.voice_cloner.generate_timed_audio(es_subs, temp_audio, self.reference_audio):
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
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
    
    def save_progress(self):
        """Save current progress to work directory"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            print(f"\n⚠️ Failed to save progress: {str(e)}")
    
    def validate_progress(self):
        """Validate that all files mentioned in progress exist"""
        if not self.progress:
            return True
        
        print("\n🔍 Validating saved progress...")
        valid_progress = {}
        
        for segment_key, data in self.progress.items():
            output_path = data.get("output")
            if not output_path or not os.path.exists(output_path):
                print(f"⚠️ Missing segment file: {segment_key}")
                continue
            
            # Check for required files
            try:
                # Handle float values in segment keys
                _, start_str, end_str = segment_key.split('_')
                start_time = int(float(start_str))
                end_time = int(float(end_str))
                
                # Check subtitle files
                ru_srt = os.path.join(self.TEMP_DIR, "subtitles", f"ru_{start_time}_{end_time}.srt")
                es_srt = os.path.join(self.TEMP_DIR, "subtitles", f"es_{start_time}_{end_time}.srt")
                
                if not os.path.exists(ru_srt) or not os.path.exists(es_srt):
                    print(f"⚠️ Missing subtitle files for segment: {segment_key}")
                    continue
                    
                # If all files exist, keep this progress
                valid_progress[segment_key] = data
                
            except (ValueError, IndexError) as e:
                print(f"⚠️ Invalid segment key format: {segment_key}")
                continue
        
        invalid_segments = len(self.progress) - len(valid_progress)
        if invalid_segments > 0:
            print(f"\n⚠️ Found {invalid_segments} invalid segments, they will be reprocessed")
            self.progress = valid_progress
            self.save_progress()
        else:
            print("✓ All progress files validated")
        
        return True
    
    def dub_video(self, input_video_path, output_video_path, voice_name="default", segment_duration=300):
        """Main dubbing process"""
        try:
            # Setup and initialize
            print("\n🚀 Starting video dubbing process...")
            if not self.TEMP_DIR:
                self.setup_work_directory()
            self.initialize_processors()
            
            # Validate existing progress
            self.validate_progress()
            
            # Now try to load existing voice samples
            if not self.voice_cloner:
                raise RuntimeError("Voice cloner not properly initialized")
                
            reference_samples = self.voice_cloner.load_voice_samples(voice_name)
            
            if reference_samples:
                print("\n🎤 Found existing voice samples")
                print("Play samples? (y/n):")
                if input().lower().strip() == 'y':
                    approved_samples = verify_voice_samples(
                        reference_samples,
                        self.voice_cloner,
                        input_video_path,
                        reuse_mode=True
                    )
                    if approved_samples:
                        reference_samples = approved_samples
                    else:
                        reference_samples = None
                else:
                    reference_samples = None
            
            # If no samples or samples were rejected, extract new ones
            if not reference_samples:
                print("\n🎤 Extracting new voice samples...")
                new_samples = self.voice_cloner.extract_voice_references(input_video_path)
                approved_samples = verify_voice_samples(
                    new_samples,
                    self.voice_cloner,
                    input_video_path
                )
                
                if not approved_samples:
                    print("\n⚠️ Process cancelled by user")
                    return
                    
                # Save approved samples for future use
                reference_samples = self.voice_cloner.save_approved_samples(approved_samples, voice_name)
            
            # Store the reference audio path for use in processing
            self.reference_audio = reference_samples[0]
            
            # Process video in segments
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
            if dubbed_segments:
                print(f"\n🔄 Preparing to concatenate {len(dubbed_segments)} segments...")
                
                # Create concat file
                concat_file = os.path.join(self.TEMP_DIR, "segments.txt")
                with open(concat_file, "w") as f:
                    for segment in dubbed_segments:
                        f.write(f"file '{os.path.abspath(segment)}'\n")
                
                # Use FFmpeg for fast concatenation
                import subprocess
                cmd = [
                    'ffmpeg', '-y',          # Overwrite output file if exists
                    '-f', 'concat',          # Use concat demuxer
                    '-safe', '0',            # Don't restrict file paths
                    '-i', concat_file,       # Input is our list file
                    '-c', 'copy',            # Stream copy (no re-encode)
                    '-movflags', '+faststart', # Optimize for web playback
                    output_video_path
                ]
                
                print("\n💾 Fast concatenating segments...")
                print("⚡ Using FFmpeg direct stream copy (should take minutes, not hours)")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Monitor FFmpeg progress
                while True:
                    output = process.stderr.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        if "time=" in output:
                            # Extract progress information
                            time_match = re.search(r"time=(\d{2}:\d{2}:\d{2})", output)
                            if time_match:
                                print(f"\rProgress: {time_match.group(1)}", end="")
                
                if process.returncode == 0:
                    print("\n\n✨ Done! Output saved to:", output_video_path)
                else:
                    raise RuntimeError("FFmpeg concatenation failed")
                
                # Cleanup concat file
                os.remove(concat_file)
                
            else:
                raise RuntimeError("No segments were successfully processed")
                
        except Exception as e:
            print(f"\n❌ Error during processing: {str(e)}")
            raise

def cleanup_handler(signum, frame):
    """Handle cleanup on interrupt"""
    print("\n\n⚠️ Process interrupted, cleaning up...")
    sys.exit(1)

if __name__ == "__main__":
    try:
        # Register cleanup handler
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        
        # Get input/output paths
        input_video = "russian_video.mp4"
        output_video = "spanish_dubbed_video.mp4"
        
        # Create dubber instance
        work_dir = f"workdir_{os.path.splitext(os.path.basename(input_video))[0]}"
        dubber = VideoDubber(work_dir=work_dir)
        
        # Start dubbing process
        dubber.dub_video(input_video, output_video, segment_duration=30)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
