import sys

# Add the path to the sys.path list
sys.path.append('/Users/soli/Desktop/Persian_Tokenizer/minbpe')
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

import arabic_reshaper
from bidi.algorithm import get_display
from termcolor import colored
import random 

llama_text = """
سنگی نفتد گوشه بام
بخت شوریده
مردیم چشمه حیوان رساند
شرح عطش سینه تفسیده
فریاد دوری برافشاند
عرصه شطرنج چیده
هجران سیلی غم کور
دل تیغ نترسیده
شعله شوق حیله نشاندیم
"""



def test_encode_decode_identity(tokenizer_factory, text):
    tokenizer = tokenizer_factory()
    # Encode text to IDs
    ids = tokenizer.encode(text)
    # Decode IDs back to text
    decoded = tokenizer.decode(ids)
    decoded_rtl = convert(decoded)

    # Assuming tokenizer is already defined and `ids` is your list of IDs
    colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

    
    # Convert original text to raw bytes
    text_ids = list(text.encode("utf-8"))
    # print(txt)
    # Check if decoded IDs match the raw byte values
    print("Decoded matches original text:", text == decoded)
    # Output the ratio of length reduction
    print("Reduction ratio (original bytes / encoded IDs):", len(text_ids) / len(ids))


    for id in ids:
        decoded_token = tokenizer.decode([id])
        color = random.choice(colors)
        print(colored(convert(decoded_token), color),end= "")

    # Decode the bytes object to a string using UTF-8 encoding


#for rtl issue 
def convert(text):
    reshaped_text = arabic_reshaper.reshape(text)
    converted = get_display(reshaped_text)
    return converted




test_encode_decode_identity(BasicTokenizer, llama_text)






