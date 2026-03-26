from __future__ import annotations

import unittest

from whisper_pularr.whisper_prompt import configure_whisper_prompt, resolve_whisper_language


class _DummyTokenizer:
    def __init__(self) -> None:
        self.lang_to_id = {"<|en|>": 1, "<|fr|>": 2}
        self.prefix_tokens = [99, 1, 2, 3]
        self.last_prefix_kwargs = None

    def set_prefix_tokens(self, **kwargs) -> None:
        self.last_prefix_kwargs = kwargs
        if "language" in kwargs:
            self.prefix_tokens = [99, self.lang_to_id[kwargs["language"]]]
        else:
            self.prefix_tokens = [99]


class _DummyGenerationConfig:
    def __init__(self) -> None:
        self.task = "translate"
        self.language = "<|en|>"
        self.return_timestamps = True
        self.forced_decoder_ids = [(1, 2)]


class _DummyConfig:
    def __init__(self) -> None:
        self.forced_decoder_ids = [(1, 2)]


class _DummyModel:
    def __init__(self) -> None:
        self.generation_config = _DummyGenerationConfig()
        self.config = _DummyConfig()


class _DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()


class WhisperPromptTests(unittest.TestCase):
    def test_resolve_whisper_language_accepts_bare_code(self) -> None:
        tokenizer = _DummyTokenizer()
        self.assertEqual(resolve_whisper_language(tokenizer, "en"), "<|en|>")

    def test_omitted_language_clears_generation_language_and_forced_ids(self) -> None:
        processor = _DummyProcessor()
        model = _DummyModel()

        configure_whisper_prompt(processor=processor, model=model, language=None)

        self.assertEqual(processor.tokenizer.last_prefix_kwargs, {"task": "transcribe"})
        self.assertEqual(processor.tokenizer.prefix_tokens, [99])
        self.assertEqual(model.generation_config.task, "transcribe")
        self.assertIsNone(model.generation_config.language)
        self.assertIsNone(model.generation_config.forced_decoder_ids)
        self.assertIsNone(model.config.forced_decoder_ids)

    def test_explicit_supported_language_is_applied(self) -> None:
        processor = _DummyProcessor()
        model = _DummyModel()

        configure_whisper_prompt(processor=processor, model=model, language="fr")

        self.assertEqual(processor.tokenizer.last_prefix_kwargs, {"task": "transcribe", "language": "<|fr|>"})
        self.assertEqual(model.generation_config.language, "<|fr|>")
        self.assertIsNone(model.generation_config.forced_decoder_ids)


if __name__ == "__main__":
    unittest.main()
