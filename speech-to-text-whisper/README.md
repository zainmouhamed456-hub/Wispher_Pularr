# Speech-to-Text Whisper

A polished, Google Colab-ready demonstration of speech transcription with `openai/whisper-small` using Hugging Face Transformers.

This project is designed for advanced users who want a minimal but production-minded Colab workflow:

- minimal dependencies
- explicit GPU and precision handling
- efficient long-form inference via the Transformers ASR pipeline
- clean upload, transcription, and output formatting flow

## Repository Structure

```text
speech-to-text-whisper/
|-- whisper_transcription.ipynb
|-- README.md
`-- requirements.txt
```

## Colab-Focused Setup

1. Upload this repository to GitHub.
2. Open `whisper_transcription.ipynb` in Google Colab.
3. In Colab, select `Runtime > Change runtime type > T4 GPU`.
4. Run the notebook from top to bottom.

The notebook installs only the packages that are not reliably provided by Colab by default. It assumes Colab already provides a CUDA-enabled PyTorch build when a GPU runtime is selected.

## Usage Guide

1. Run the environment and dependency cells.
2. Load the Whisper small model from Hugging Face.
3. Upload an audio file with the provided Colab upload cell.
4. Run the transcription cell.
5. Review the formatted transcript and download the saved output files if needed.

The notebook saves outputs to:

- `outputs/transcription.txt`
- `outputs/transcription.json`

## GPU (T4) Usage Notes

- On a Colab T4 GPU, the notebook automatically switches inference to `FP16`.
- On CPU-only runtimes, it falls back to `FP32`.
- The notebook uses the Transformers `automatic-speech-recognition` pipeline with `chunk_length_s=30`, which is the recommended path for long-form Whisper inference on the Hugging Face side.
- `openai/whisper-small` is a multilingual checkpoint, so the notebook defaults to `task="transcribe"` and leaves language detection automatic unless you explicitly set `LANGUAGE`.

## Example Output

```text
Transcription Summary
- File: sample.wav
- Model: openai/whisper-small
- Device: cuda:0
- Precision: FP16
- Task: transcribe
- Language: auto
- Elapsed Time: 2.41 seconds

Transcript
Hello everyone, and welcome to this Whisper transcription demo.
```

## References

- Hugging Face Whisper docs: https://huggingface.co/docs/transformers/en/model_doc/whisper
- Hugging Face model card for `openai/whisper-small`: https://huggingface.co/openai/whisper-small
- Google Colab FAQ: https://research.google.com/colaboratory/intl/en-GB/faq.html
