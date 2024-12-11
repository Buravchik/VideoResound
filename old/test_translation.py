import os
import tempfile
import whisper
import torch
from gtts import gTTS
import shutil
from googletrans import Translator

# Global variables for cleanup
TEMP_DIR = None
CLEANUP_FILES = []

def setup_temp_directory():
    """Create a temporary directory for all intermediate files"""
    global TEMP_DIR
    TEMP_DIR = tempfile.mkdtemp(prefix="translation_test_")
    print(f"📁 Created temporary directory: {TEMP_DIR}")
    return TEMP_DIR

def cleanup():
    """Clean up all temporary files and directory"""
    global TEMP_DIR, CLEANUP_FILES
    
    print("\n🧹 Cleaning up temporary files...")
    for file in CLEANUP_FILES:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"✓ Removed: {file}")
        except Exception as e:
            print(f"⚠️ Failed to remove {file}: {str(e)}")
    
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"✓ Removed temporary directory: {TEMP_DIR}")
        except Exception as e:
            print(f"⚠️ Failed to remove temporary directory: {str(e)}")
    
    CLEANUP_FILES = []
    TEMP_DIR = None

class TranslationProcessor:
    def __init__(self):
        self.translator = Translator()
        self.whisper_model = whisper.load_model("medium")
        print(f"✓ Initialized translation processor")
    
    def process_text(self, audio_path):
        """Process audio: transcribe Russian -> translate to Spanish"""
        try:
            # First transcribe using Whisper (what it's good at)
            transcription = self.whisper_model.transcribe(
                audio_path,
                language="ru",
                task="transcribe",
                fp16=False
            )
            
            russian_text = transcription['text'].strip()
            print(f"🎤 Transcribed: {russian_text}")
            
            # Then translate using Google Translate (what it's good at)
            translation = self.translator.translate(
                russian_text,
                src='ru',
                dest='es'
            )
            
            return {
                'transcribed': russian_text,
                'translated': translation.text
            }
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return None

def test_translation():
    """Test translation pipeline with sample Russian text"""
    processor = TranslationProcessor()
    
    test_samples = [
        "Привет, как дела?",  # Should be: "¡Hola! ¿Cómo estás?"
        "Сегодня прекрасная погода",  # Should be: "Hoy hace un tiempo hermoso"
        "Я люблю программировать на Python",  # Should be: "Me encanta programar en Python"
        "Искусственный интеллект меняет мир",  # Should be: "La inteligencia artificial está cambiando el mundo"
        "Это тестовый пример для проверки перевода",  # Should be: "Este es un ejemplo de prueba para verificar la traducción"
        "Доброе утро",  # Should be: "Buenos días"
        "Спасибо за помощь",  # Should be: "Gracias por la ayuda"
        "Как тебя зовут?",  # Should be: "¿Cómo te llamas?"
    ]
    
    print("\n🔄 Testing Translation Pipeline:")
    print("=" * 80)
    print(f"{'Test #':<8}{'Russian Text':<40}{'Spanish Translation':<40}")
    print("=" * 80)
    
    results = []
    for i, text in enumerate(test_samples, 1):
        try:
            temp_audio = os.path.join(TEMP_DIR, f"test_audio_{i}.wav")
            CLEANUP_FILES.append(temp_audio)
            
            # Create audio from text using gTTS
            tts = gTTS(text=text, lang='ru', slow=False)
            tts.save(temp_audio)
            
            # Process using the pipeline
            print(f"\n📝 Processing test {i}...")
            result = processor.process_text(temp_audio)
            
            if result:
                results.append({
                    'test_num': i,
                    'original': text,
                    'transcribed': result['transcribed'],
                    'translated': result['translated']
                })
            else:
                results.append({
                    'test_num': i,
                    'original': text,
                    'transcribed': "❌ Failed",
                    'translated': "❌ Failed"
                })
            
        except Exception as e:
            print(f"⚠️ Error in test {i}: {str(e)}")
            results.append({
                'test_num': i,
                'original': text,
                'transcribed': f"❌ Error",
                'translated': f"❌ Error: {str(e)}"
            })
    
    # Print final results table
    print("\n📊 Results:")
    print("=" * 120)
    print(f"{'Test #':<8}{'Original':<30}{'Transcribed':<30}{'Translated':<30}")
    print("-" * 120)
    
    for result in results:
        test_num = f"#{result['test_num']}"
        original = result['original']
        transcribed = result['transcribed']
        translated = result['translated']
        
        # Truncate long strings
        if len(original) > 27: original = original[:24] + "..."
        if len(transcribed) > 27: transcribed = transcribed[:24] + "..."
        if len(translated) > 27: translated = translated[:24] + "..."
            
        print(f"{test_num:<8}{original:<30}{transcribed:<30}{translated:<30}")
    
    print("=" * 120)
    
    # Print statistics
    success_count = sum(1 for r in results if not r['translated'].startswith('❌'))
    print(f"\n📈 Statistics:")
    print(f"Total tests: {len(results)}")
    print(f"Successful translations: {success_count}")
    print(f"Failed translations: {len(results) - success_count}")
    print(f"Success rate: {(success_count/len(results))*100:.1f}%")

if __name__ == "__main__":
    try:
        print("\n🎯 Starting Translation Test")
        print("=" * 80)
        
        TEMP_DIR = setup_temp_directory()
        test_translation()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
    finally:
        print("\n🧹 Cleaning up...")
        cleanup()
        print("\n✨ Test completed!")