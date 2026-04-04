# Canvas: test_text_sanitizer_v3
# Path: tests/features/domain_phonetic_correction/utils/test_text_sanitizer.py

from app.features.domain_phonetic_correction.utils.text_sanitizer import PhoneticTextSanitizer

class TestPhoneticTextSanitizer:
    def test_removes_trailing_zero_artifacts(self):
        raw_text = "trade that drill0 So0 onwards0 and0 I'll0 explain0 the0"
        clean_text = PhoneticTextSanitizer.sanitize(raw_text)
        assert clean_text == "trade that drill So onwards and I'll explain the"

    def test_removes_leading_zero_artifacts(self):
        raw_text = "0the 0market's now getting a 0little bit stretched0"
        clean_text = PhoneticTextSanitizer.sanitize(raw_text)
        assert clean_text == "the market's now getting a little bit stretched"

    def test_preserves_valid_numbers_and_years(self):
        # We DO NOT want to delete the '0' in 1440, $1,000, 2018, or 2.0
        raw_text = "In 2018 the Bund hit 1440 and cost $1,000 for version 2.0"
        clean_text = PhoneticTextSanitizer.sanitize(raw_text)
        assert clean_text == "In 2018 the Bund hit 1440 and cost $1,000 for version 2.0"

    def test_preserves_valid_alphanumeric_acronyms(self):
        # We DO NOT want to destroy legitimate model names or codes like "B2B" or "S0"
        raw_text = "Speaker S0 is discussing B2B sales."
        clean_text = PhoneticTextSanitizer.sanitize(raw_text)
        assert clean_text == "Speaker S0 is discussing B2B sales."