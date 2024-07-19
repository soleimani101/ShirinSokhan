import pytest
import tiktoken
import os
import sys
import time

# Add the path to the custom tokenizer module
sys.path.append('/Users/soli/Desktop/Persian_Tokenizer/minbpe')
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

# Directory containing text files

# -----------------------------------------------------------------------------
# common test data

# a few strings to test the tokenizers on
test_strings = [
    "", # empty string
    "؟", # single character (Persian question mark)
    "سلام دنیا!!!؟ (سلام!) لول123 😉", # fun small string in Persian
]

def unpack(text):
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        example_file = os.path.join(dirname, text[5:])
        contents = open(example_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

specials_string = """
سلام دنیا این یک سند است
و این یک سند دیگر است
و این یکی نشانه گذاری دارد. FIM
آخرین سند!!! 👋
""".strip()


llama_text = """
یک متن فارسی برای تست
""".strip()

# -----------------------------------------------------------------------------
# tests

# test encode/decode identity for a few different strings
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer, GPT4Tokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded






# test that our tokenizer matches the official GPT-4 tokenizer

# reference test to add more tests in the future
# @pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
# def test_persian_example(tokenizer_factory):
#     """
#     Quick unit test for Persian text.
#     """
#     tokenizer = tokenizer_factory()
#     text = "سلام دنیا سلام دنیا"
#     tokenizer.train(text, 256 + 3)
#     ids = tokenizer.encode(text)
#     # Provide the expected ids after BPE merge. This part will depend on the actual merges.
#     expected_ids = [256, 257, 256, 257] # This is just an example, adjust according to your BPE merge results.
#     assert ids == expected_ids
#     assert tokenizer.decode(tokenizer.encode(text)) == text


if __name__ == "__main__":
    pytest.main()
