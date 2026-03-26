from __future__ import annotations

import unittest

from whisper_pularr.text import normalize_transcript


class TextNormalizationTests(unittest.TestCase):
    def test_normalizes_apostrophes_hyphens_and_spacing(self) -> None:
        self.assertEqual(
            normalize_transcript("Mi  yidi  ɗo\u2019on  e leyde\u2011mum !"),
            "mi yidi ɗo'on e leyde-mum",
        )

    def test_drops_outer_punctuation_but_keeps_internal_word_marks(self) -> None:
        self.assertEqual(
            normalize_transcript("« Centre d' expension rural », walla ?"),
            "centre d'expension rural walla",
        )

    def test_mixed_french_loanwords_are_normalized_consistently(self) -> None:
        self.assertEqual(
            normalize_transcript("Ministère_de_l’ agriculture au Sénégal."),
            "ministère de l'agriculture au sénégal",
        )


if __name__ == "__main__":
    unittest.main()
