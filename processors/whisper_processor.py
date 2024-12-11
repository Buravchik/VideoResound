import os
import moviepy.editor as mp
from models.subtitle import SubtitleSegment
from googletrans import Translator
import json

class WhisperProcessor:
    def __init__(self, model_size="medium"):
        print(f"\n🎯 Initializing Whisper ({model_size})...")
        import whisper
        self.model = whisper.load_model(model_size)
        self.translator = Translator()
        self.translation_cache = {}
        self.cache_file = None
        print("✓ Whisper model loaded")
        print("✓ Translator initialized")
        
    def process_video_segment(self, video_path, start_time, end_time, temp_dir):
        """Process video segment and return Russian and Spanish subtitles"""
        try:
            # Extract audio from video segment
            video = mp.VideoFileClip(video_path).subclip(start_time, end_time)
            temp_audio = os.path.join(temp_dir, f"temp_whisper_{start_time}_{end_time}.wav")
            video.audio.write_audiofile(temp_audio, logger=None)
            video.close()
            
            # Transcribe with Whisper
            print("\n🎯 Transcribing audio...")
            result = self.model.transcribe(
                temp_audio,
                language="ru",
                task="transcribe",
                fp16=False
            )
            
            # Clean up temp audio
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            ru_subs = []
            es_subs = []
            
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                ru_text = segment.get("text", "").strip()
                
                if ru_text:
                    es_text = self.translate_to_spanish(ru_text)
                    ru_subs.append(SubtitleSegment(start, end, ru_text))
                    es_subs.append(SubtitleSegment(start, end, es_text))
            
            # Save subtitles to files
            if ru_subs:
                ru_srt = os.path.join(temp_dir, "subtitles", f"ru_{start_time}_{end_time}.srt")
                es_srt = os.path.join(temp_dir, "subtitles", f"es_{start_time}_{end_time}.srt")
                
                self.save_subtitles(ru_subs, ru_srt)
                self.save_subtitles(es_subs, es_srt)
                
            return ru_subs, es_subs
            
        except Exception as e:
            print(f"❌ Error processing video segment: {str(e)}")
            return [], []
    
    def load_translation_cache(self, work_dir):
        """Load translation cache from work directory"""
        self.cache_file = os.path.join(work_dir, "translation_cache.json")
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.translation_cache = json.load(f)
                print(f"✓ Loaded {len(self.translation_cache)} cached translations")
            except Exception as e:
                print(f"⚠️ Failed to load translation cache: {str(e)}")
                self.translation_cache = {}

    def save_translation_cache(self):
        """Save translation cache to work directory"""
        if self.cache_file and self.translation_cache:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ Failed to save translation cache: {str(e)}")

    def translate_to_spanish(self, text):
        """Translate text from Russian to Spanish using Google Translate with caching"""
        if not text:
            return text

        # Check cache first
        if text in self.translation_cache:
            return self.translation_cache[text]

        try:
            translation = self.translator.translate(text, src='ru', dest='es')
            translated = translation.text.strip()
            if translated:
                # Cache the successful translation
                self.translation_cache[text] = translated
                if self.cache_file:
                    self.save_translation_cache()
                return translated
            print(f"⚠️ Translation failed for: {text}")
            return text
        except Exception as e:
            print(f"⚠️ Translation error: {str(e)}")
            return text

    def save_subtitles(self, subs, output_path):
        """Save subtitles to SRT file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, sub in enumerate(subs, 1):
                    start_time = self.format_timestamp(sub.start)
                    end_time = self.format_timestamp(sub.end)
                    f.write(f"{i}\n{start_time} --> {end_time}\n{sub.content}\n\n")
        except Exception as e:
            print(f"⚠️ Failed to save subtitles to {output_path}: {str(e)}")

    def format_timestamp(self, seconds):
        """Format seconds to SRT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
