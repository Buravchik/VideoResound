class SubtitleSegment:
    def __init__(self, start, end, content):
        self.start = start
        self.end = end
        self.content = content.strip()
        
    def __str__(self):
        return f"{self.start:.1f} -> {self.end:.1f}: {self.content}"
