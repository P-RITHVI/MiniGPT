from scripts.prepare_data import read_local_inputs


def test_combine_plain_and_gutenberg_inputs(tmp_path):
    plain = tmp_path / "plain.txt"
    gutenberg = tmp_path / "gutenberg.txt"
    plain.write_text("First book.", encoding="utf-8")
    gutenberg.write_text("Header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nSecond book.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nFooter", encoding="utf-8")
    assert read_local_inputs([plain, gutenberg]) == "First book.\n\nSecond book."
