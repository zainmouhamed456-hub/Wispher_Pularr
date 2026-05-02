from __future__ import annotations

from fairseq2 import __version__ as fairseq2_version
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def main() -> None:
    print("Omnilingual ASR import OK")
    print(f"fairseq2 {fairseq2_version}")
    print(f"ASR pipeline class: {ASRInferencePipeline.__name__}")


if __name__ == "__main__":
    main()
