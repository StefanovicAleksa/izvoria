# Canvas: test_xml_parser_v3
# Path: tests/features/domain_phonetic_correction/utils/test_xml_parser.py

from app.features.domain_phonetic_correction.utils.xml_parser import parse_xml_patches


class TestXMLParser:
    def test_parse_standard_xml_patches(self):
        raw_output = """
        <patch>
            <orig>with a cute angina</orig>
            <corr>with acute angina</corr>
        </patch>
        """
        patches = parse_xml_patches(raw_output)

        assert len(patches) == 1
        assert patches[0].original == "with a cute angina"
        assert patches[0].correction == "with acute angina"

    def test_parse_blocks_nested_xml_leakage(self):
        # NEW TEST: Proves the [^<]*? regex firewall stops XML from leaking into the transcript
        malicious_output = """
        <patch>
            <orig>some bad text</orig>
            <corr>some <patch>good</patch> text</corr>
        </patch>
        """
        patches = parse_xml_patches(malicious_output)

        # Because the <corr> tag contains `<` and `>`, the Regex safely refuses to parse it.
        assert len(patches) == 0

    def test_parse_multiple_patches_with_hallucinated_noise(self):
        raw_output = """
        Sure, here are the corrections for the transcript!

        <patch>
            <orig>eat bit da margin</orig>
            <corr>EBITDA margin</corr>
        </patch>

        I hope this helps!

        <patch><orig>syn err G</orig><corr>synergy</corr></patch>
        """
        patches = parse_xml_patches(raw_output)

        assert len(patches) == 2
        assert patches[0].original == "eat bit da margin"
        assert patches[1].correction == "synergy"

    def test_parse_empty_string_returns_empty_list(self):
        patches = parse_xml_patches("Looks perfectly fine to me!")
        assert len(patches) == 0