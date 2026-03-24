from faster_whisper import WhisperModel
import time

print("--- INITIALIZING FASTER-WHISPER ---")
print("Loading the 'base' model into the RTX 5060...")

try:
    start_time = time.time()
    
    # Initialize the model on the GPU using 16-bit precision
    model = WhisperModel("base", device="cuda", compute_type="float16")
    
    load_time = time.time() - start_time
    print(f"\nSUCCESS: Model loaded into VRAM in {load_time:.2f} seconds!")
    print("Your GPU is ready to transcribe audio.")

except Exception as e:
    print("\nERROR: Failed to load into GPU.")
    print(str(e))