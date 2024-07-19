
from .base import Tokenizer, get_stats, merge
import pickle

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.merges = {}  # Used in encode() and decode()
        self.vocab = {}   # Used in encode() and decode()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256

        # Input text preprocessing
        text_bytes = text.encode("utf-8")  # Raw bytes
        ids = list(text_bytes)  # List of integers in range 0..255

        # Initialize merge mappings
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        next_idx = 256

        while True:
            # Count the number of occurrences of each consecutive pair
            stats = get_stats(ids)
            if not stats:
                break  # No more pairs to merge

            # Find the pair with the highest count
            pair = max(stats, key=stats.get)
            if stats[pair] == 0:
                break  # No pair has a count > 0, stop merging

            # Mint a new token and assign it the next available index
            if next_idx >= vocab_size:
                break  # Reached the vocabulary size limit
            idx = next_idx
            next_idx += 1

            # Replace all occurrences of the pair in ids with idx
            ids = merge(ids, pair, idx)

            # Save the merge and vocabulary
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # if verbose:
            #     print(f"Merge {next_idx-256}/{vocab_size-256}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # Save class variables
        self.merges = merges
        self.vocab = vocab




    def decode(self, ids):
        # Load merges from file if not already set
        if not self.merges:
            self.load_merges()

        # Create a dictionary to map ID to its replacement
        id_to_replacement = {v: k for k, v in self.merges.items() if isinstance(v, int)}

        def replace_id(id):
            # Replace with the tuple key from merges if the id is greater than 256
            return id_to_replacement.get(id, id)

        def apply_replacements(ids):
            new_ids = []
            for id in ids:
                if isinstance(id, tuple):
                    new_ids.extend(replace_id(sub_id) for sub_id in id)
                else:
                    new_ids.append(replace_id(id))
            return new_ids

        # Initialize the list of IDs to process
        current_ids = ids

        while any(isinstance(id, int) and id > 256 for id in current_ids):
            # Apply replacements
            current_ids = apply_replacements(current_ids)
            # Flatten the list (if needed) and ensure all IDs are processed
            flattened_ids = []
            for item in current_ids:
                if isinstance(item, tuple):
                    flattened_ids.extend(item)
                else:
                    flattened_ids.append(item)
            current_ids = flattened_ids



        byte_data = bytes(current_ids)
    # Decode the bytes object to a string using UTF-8 encoding
        decoded_text = byte_data.decode("utf-8")

        return decoded_text

    def encode(self, text):
        # Load merges from file if not already set
        if not self.merges:
            self.load_merges()

        # Given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # Raw bytes
        ids = list(text_bytes)  # List of integers in range 0..255

        while len(ids) >= 2:
            # Find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # No more merges available
            # Merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def load_merges(self):
        with open('models/basic.model', 'r') as f:
            lines = f.readlines()

        merge_index = 256
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    key1, key2 = map(int, parts)
                    self.merges[(key1, key2)] = merge_index
                    merge_index += 1
                except ValueError:
                    print(f"Skipping line with invalid data: {line}")
            else:
                print(f"Skipping line with unexpected format: {line}")
