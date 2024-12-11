from video_dubbing import dub_video

if __name__ == "__main__":
    input_video = input("Enter path to Russian video file: ")
    output_video = input("Enter path for output Spanish video (or press Enter for default): ") or "spanish_output.mp4"
    
    print("\nStarting video dubbing process...")
    dub_video(input_video, output_video, segment_duration=300) 