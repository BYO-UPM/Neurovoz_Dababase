import os
import whisper
from tqdm import tqdm


def transcribe_audio(directory):
    model = whisper.load_model("large")  # You can adjust the model size as needed
    files = [f for f in os.listdir(directory) if f.endswith(".wav")]

    # Ensure the output directory exists
    check_dir = "./data/version_to_zenodo/transcriptions"
    output_dir = "./data/version_to_zenodo/transcriptions_new"
    os.makedirs(output_dir, exist_ok=True)

    for file in tqdm(files):
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.txt")
        check_dir_path = os.path.join(check_dir, f"{os.path.splitext(file)[0]}.txt")

        # Check if the transcription already exists
        if os.path.exists(check_dir_path):
            print(f"Transcription for {file} already exists, skipping...")
            continue

        file_path = os.path.join(directory, file)
        print(f"Transcribing {file}...")
        try:
            result = model.transcribe(file_path, language="es")
        except Exception as e:
            print(f"Error transcribing {file}: {e}")
            continue
        transcription = result["text"]

        # Save the transcription
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            out_file.write(transcription)

        # Save the audio used in the same folder
        os.system(f"cp {file_path} {output_dir}")

        print(f"Transcription for {file} saved to {output_file_path}")


if __name__ == "__main__":
    directory = (
        "data/version_to_zenodo/audios"  # Update this to the path of your audio folder
    )
    transcribe_audio(directory)


# # Check if there exists a transcription for all the audios

# audio_path = "data/version_to_zenodo/audios"

# audio_files = [f.split(".")[0] for f in os.listdir(audio_path) if f.endswith(".wav")]

# tran_path = "data/version_to_zenodo/transcriptions"

# tran_files = [f.split(".")[0] for f in os.listdir(tran_path) if f.endswith(".txt")]

# # Get the diff
# diff = set(audio_files) - set(tran_files)
