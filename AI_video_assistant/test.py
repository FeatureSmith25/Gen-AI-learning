from utils.audio_processor import process_input
from Core.transcriber import transcribe_all

source= "https://www.youtube.com/watch?v=bj6XIBb9xLU"

chunks=process_input(source)

print(transcribe_all(chunks))