from TTS.api import TTS
import os

def test_xtts():
    try:
        print("🎙️ Testing XTTS v2 initialization...")
        print("\nChecking package versions...")
        import pydantic
        print(f"Pydantic version: {pydantic.__version__}")
        
        # Initialize TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("✓ Model loaded successfully")
        
        # Create a test directory if it doesn't exist
        os.makedirs("test_audio", exist_ok=True)
        
        # First generate a reference audio using a simple model
        print("\nGenerating reference audio...")
        simple_tts = TTS("tts_models/en/ljspeech/tacotron2-DCA")
        simple_tts.tts_to_file(
            text="This is a reference audio sample.", 
            file_path="test_audio/reference.wav"
        )
        print("✓ Reference audio generated")
        
        # Now test XTTS with the reference audio
        print("\nTesting XTTS with reference audio...")
        text = "This is a test for XTTS v2."
        tts.tts_to_file(
            text=text,
            file_path="test_audio/xtts_output.wav",
            speaker_wav="test_audio/reference.wav",
            language="en"
        )
        print("✓ Voice generation successful")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_xtts() 