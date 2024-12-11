# test.py
try:
    import moviepy
    print("1. Base moviepy import successful")
    print(f"MoviePy location: {moviepy.__file__}")
    
    import moviepy.editor
    print("2. moviepy.editor import successful")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()