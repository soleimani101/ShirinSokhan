import os
import time
from minbpe import BasicTokenizer, RegexTokenizer
from tqdm import tqdm

# Directory containing text files
# text_files_dir = "dataTrain_hight_vol"
text_files_dir = "dataTrain_low_vol"

# Create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):
    # Construct the Tokenizer object once
    tokenizer = TokenizerClass()
    
    # List all text files
    text_files = [f for f in os.listdir(text_files_dir) if f.endswith(".txt")]
    
    # Loop through each file and tokenize individually with a progress bar
    for filename in tqdm(text_files, desc=f"Processing files with {name} tokenizer"):
        filepath = os.path.join(text_files_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        
        # Train the tokenizer on the individual file content
        tokenizer.train(text, 5000, verbose=True)
    
    # Save the tokenizer model after processing all files
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
