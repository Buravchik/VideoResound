import os
import time
import json
import random
import shutil
from tqdm import tqdm
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import moviepy.editor as mp

class VoiceCloner:
    def __init__(self):
        print("\n🎙️ Initializing TTS model...")
        try:
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self.reference_samples = []
            self.used_segments = set()
            self.samples_dir = None  # Will be set when work_dir is provided
            self.extracted_dir = None
            self.approved_dir = None
            print("✓ Model loaded")
        except Exception as e:
            print(f"Detailed error: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Failed to initialize TTS model: {str(e)}")

    def set_work_directory(self, work_dir):
        """Set up voice references directory in work directory"""
        self.samples_dir = os.path.join(work_dir, "voice_references")
        self.extracted_dir = os.path.join(self.samples_dir, "extracted")
        self.approved_dir = os.path.join(self.samples_dir, "approved")
        os.makedirs(self.extracted_dir, exist_ok=True)
        os.makedirs(self.approved_dir, exist_ok=True)

    def extract_voice_references(self, video_path, min_duration=3, max_duration=10, silence_thresh=-40):
        """Extract potential voice samples from video"""
        try:
            print("\n🎤 Extracting voice samples...")
            # Load video audio
            video = mp.VideoFileClip(video_path)
            temp_audio = os.path.join(self.samples_dir, "temp_full_audio.wav")
            video.audio.write_audiofile(temp_audio, logger=None)
            video.close()
            
            # Process audio
            audio = AudioSegment.from_wav(temp_audio)
            
            # Find non-silent chunks
            print("🔍 Analyzing audio...")
            chunks = detect_nonsilent(
                audio,
                min_silence_len=500,
                silence_thresh=silence_thresh
            )
            
            extracted_samples = []
            with tqdm(total=min(len(chunks), 5), desc="Extracting samples") as pbar:
                for i, (start, end) in enumerate(chunks):
                    duration = (end - start) / 1000  # Convert to seconds
                    
                    if min_duration <= duration <= max_duration:
                        sample_path = os.path.join(self.extracted_dir, f"sample_{i+1}.wav")
                        chunk = audio[start:end]
                        chunk.export(sample_path, format="wav")
                        extracted_samples.append(sample_path)
                        pbar.update(1)
                        
                        if len(extracted_samples) >= 5:  # Limit to 5 samples
                            break
            
            # Cleanup
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            if not extracted_samples:
                print("\n⚠️ No suitable voice samples found. Trying with different parameters...")
                # Try again with more lenient parameters
                if silence_thresh == -40:
                    return self.extract_voice_references(video_path, min_duration=2, max_duration=15, silence_thresh=-45)
                elif silence_thresh == -45:
                    return self.extract_voice_references(video_path, min_duration=2, max_duration=20, silence_thresh=-50)
            
            print(f"\n✓ Extracted {len(extracted_samples)} potential voice samples")
            return extracted_samples
            
        except Exception as e:
            print(f"❌ Error extracting voice samples: {str(e)}")
            return []

    def load_voice_samples(self, voice_name="default"):
        """Load approved voice samples from work directory"""
        if not self.samples_dir:
            raise RuntimeError("Work directory not set")
        
        voice_dir = os.path.join(self.approved_dir, voice_name)
        metadata_path = os.path.join(voice_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                samples = metadata["samples"]
                if all(os.path.exists(s) for s in samples):
                    print(f"\n✓ Loaded {len(samples)} approved samples for voice '{voice_name}'")
                    return samples
        return None

    def save_approved_samples(self, approved_samples, voice_name="default"):
        """Save approved samples to work directory"""
        if not self.samples_dir:
            raise RuntimeError("Work directory not set")
        
        voice_dir = os.path.join(self.approved_dir, voice_name)
        os.makedirs(voice_dir, exist_ok=True)
        
        # Save samples with metadata
        saved_paths = []
        for i, sample_path in enumerate(approved_samples):
            new_path = os.path.join(voice_dir, f"reference_{i+1}.wav")
            if os.path.abspath(sample_path) != os.path.abspath(new_path):
                shutil.copy2(sample_path, new_path)
            saved_paths.append(new_path)
        
        # Save metadata
        with open(os.path.join(voice_dir, "metadata.json"), "w") as f:
            json.dump({
                "voice_name": voice_name,
                "samples": saved_paths,
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        
        print(f"\n✓ Saved {len(saved_paths)} approved samples for voice '{voice_name}'")
        return saved_paths

    def generate_timed_audio(self, subtitles, output_path, reference_audio_path, max_gap=1.0):
        """Generate dubbed audio with proper timing"""
        try:
            print("\n🎵 Generating dubbed audio...")
            final_audio = AudioSegment.silent(duration=0)
            last_end = 0
            
            # Verify reference audio exists in approved directory
            if not os.path.exists(reference_audio_path):
                approved_path = os.path.join(self.approved_dir, "default", os.path.basename(reference_audio_path))
                if os.path.exists(approved_path):
                    reference_audio_path = approved_path
                else:
                    raise RuntimeError(f"Reference audio not found in approved samples")
            
            sorted_subs = sorted(subtitles, key=lambda x: x.start)
            
            for i, sub in enumerate(sorted_subs):
                try:
                    start_time = sub.start
                    end_time = sub.end
                    gap = start_time - last_end if i > 0 else 0
                    
                    if gap > 0:
                        silence_duration = min(gap * 1000, max_gap * 1000)
                        final_audio += AudioSegment.silent(duration=silence_duration)
                    
                    temp_speech = f"temp_speech_{i}.wav"
                    
                    # Generate speech for subtitle
                    print(f" > Text to synthesize: {sub.content}")
                    sentences = self.split_into_sentences(sub.content)
                    print(" > Text splitted to sentences.")
                    print(sentences)
                    
                    self.tts.tts_to_file(
                        text=sub.content,
                        file_path=temp_speech,
                        speaker_wav=reference_audio_path,
                        language="es"
                    )
                    
                    if os.path.exists(temp_speech):
                        speech_segment = AudioSegment.from_wav(temp_speech)
                        final_audio += speech_segment
                        os.remove(temp_speech)
                        last_end = end_time
                    else:
                        print(f"⚠️ Failed to generate audio for: {sub.content}")
                    
                except Exception as e:
                    print(f"⚠️ Error processing subtitle {i}: {str(e)}")
                    continue  # Skip problematic subtitle and continue with next
            
            if len(final_audio) > 0:
                final_audio.export(output_path, format="wav")
                return True
            else:
                raise RuntimeError("No audio was generated")
            
        except Exception as e:
            print(f"❌ Error generating timed audio: {str(e)}")
            return False

    def _segments_overlap(self, range1, range2):
        """Check if two time ranges overlap"""
        start1, end1 = range1
        start2, end2 = range2
        return not (end1 < start2 or start1 > end2)

    def split_into_sentences(self, text):
        """Split text into sentences for better TTS processing"""
        # Simple sentence splitting on punctuation
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in '.!?':
                sentences.append(current.strip())
                current = ""
        if current:
            sentences.append(current.strip())
        return sentences