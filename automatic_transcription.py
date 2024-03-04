import os
import whisper
from tqdm import tqdm


def transcribe_audio(directory):
    model = whisper.load_model("large")  # You can adjust the model size as needed
    files = [
        f for f in os.listdir(directory) if f.endswith(".wav") and "ESPONTANEA" in f
    ]

    # Ensure the output directory exists
    output_dir = "./data/transcriptions"
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(files):
        file_path = os.path.join(directory, file)
        print(f"Transcribing {file}...")

        result = model.transcribe(file_path, language="es")
        transcription = result["text"]

        # Generate output file path
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.txt")

        # Save the transcription
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            out_file.write(transcription)

        print(f"Transcription for {file} saved to {output_file_path}")


if __name__ == "__main__":
    directory = "data/audios"  # Update this to the path of your audio folder
    transcribe_audio(directory)
