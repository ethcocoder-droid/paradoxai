import numpy as np

class SimpleTokenizer:
    """A very basic tokenizer for demonstration purposes."""
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3,
        }
        self.id_to_word = {v: k for k, v in self.vocab.items()}
        self.next_id = len(self.vocab)

    def build_vocabulary(self, texts):
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.next_id
                    self.id_to_word[self.next_id] = word
                    self.next_id += 1

    def _tokenize(self, text):
        # Simple whitespace tokenizer, convert to lowercase
        return text.lower().split()

    def encode(self, text):
        return [self.vocab.get(word, self.vocab["<unk>"]) for word in self._tokenize(text)]

    def decode(self, token_ids):
        return " ".join([self.id_to_word.get(id, "<unk>") for id in token_ids])

    def get_vocab_size(self):
        return len(self.vocab)

class TextDataset:
    """
    A dataset class for handling large text files, tokenizing them,
    and preparing them for language modeling training.
    """
    def __init__(self, filepath, tokenizer, seq_len, chunk_size=1024 * 1024): # 1MB chunk size
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self._build_vocabulary_from_file()
        self.inputs, self.targets = self._load_and_process_data()

    def _build_vocabulary_from_file(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                self.tokenizer.build_vocabulary([chunk])

    def _load_and_process_data(self):
        encoded_text = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                encoded_text.extend(self.tokenizer.encode(chunk))
        
        inputs = []
        targets = []
        if len(encoded_text) > self.seq_len:
            for i in range(len(encoded_text) - self.seq_len):
                inputs.append(encoded_text[i : i + self.seq_len])
                targets.append(encoded_text[i + 1 : i + self.seq_len + 1])
            
        return np.array(inputs), np.array(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]