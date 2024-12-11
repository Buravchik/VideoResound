import os
import sys
import time
from pydub import AudioSegment
from pydub.playback import play

def play_audio(file_path):
    """Play audio file across different platforms"""
    try:
        if sys.platform == "darwin":  # macOS
            os.system(f"afplay {file_path}")
        elif sys.platform == "win32":  # Windows
            os.system(f"start {file_path}")
        else:  # Linux/Other
            os.system(f"aplay {file_path}")
    except Exception as e:
        print(f"⚠️ Error playing audio: {e}")

def verify_voice_samples(samples, voice_cloner, input_video_path, reuse_mode=False):
    """Interactive voice sample verification for individual samples"""
    approved_samples = []
    current_samples = samples
    
    if reuse_mode:
        print("\n🎧 Verifying saved voice samples")
    else:
        print("\n🎧 Verifying new voice samples")
    
    while len(approved_samples) < 1:  # Need at least one good sample
        print("\nListen to each sample and decide (y/n)")
        
        for i, sample in enumerate(current_samples, 1):
            print(f"\n▶️ Sample {i}:")
            play_audio(sample)
            
            if reuse_mode:
                choice = input(f"Keep using sample {i}? (y/n/s=skip): ").lower().strip()
                if choice == 's':
                    print(f"⏩ Skipping verification of remaining samples")
                    # If skipping, approve all remaining samples
                    approved_samples.extend(current_samples[i-1:])
                    break
            else:
                choice = input(f"Use sample {i}? (y/n): ").lower().strip()
            
            if choice == 'y':
                approved_samples.append(sample)
                print(f"✓ Sample {i} approved")
            elif choice == 'n' and not reuse_mode:
                print(f"✗ Sample {i} rejected")
            elif choice not in ['y', 'n', 's']:
                print("❌ Invalid choice")
                continue
        
        if not approved_samples:
            if reuse_mode:
                print("\n⚠️ No saved samples approved. Switching to new sample extraction.")
                return None
            
            print("\n⚠️ No samples approved. Need at least one good sample.")
            choice = input("Extract new samples? (y/n/q): ").lower().strip()
            
            if choice == 'q':
                return None
            elif choice == 'y':
                print("\n🔄 Extracting new samples...")
                current_samples = voice_cloner.extract_voice_references(
                    input_video_path,
                    min_duration=4,
                    max_duration=8,
                    silence_thresh=-45
                )
            else:
                return None
    
    print(f"\n✓ {len(approved_samples)} samples approved")
    return approved_samples

def merge_audio_segments(segments, output_path, max_gap=1.0):
    """Merge multiple audio segments with controlled gaps"""
    try:
        final_audio = AudioSegment.empty()
        last_end = 0
        
        for segment in segments:
            gap = segment.start - last_end if last_end > 0 else 0
            if gap > 0:
                silence_duration = min(gap * 1000, max_gap * 1000)
                final_audio += AudioSegment.silent(duration=silence_duration)
            
            final_audio += segment.audio
            last_end = segment.end
        
        final_audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"❌ Error merging audio segments: {str(e)}")
        return False
