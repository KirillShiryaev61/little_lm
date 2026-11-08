"""Реализация простого токенайзера по словам и знакам препинания."""

import re

class SimpleTokenizerV1():

    def __init__(self, vocab):
        self.char_to_id = dict(vocab)

        self.special_tokens = ["<PAD>", "<UNK>"]
        start_idx = len(self.char_to_id)

        for token in self.special_tokens:
            if token not in self.char_to_id:
                self.char_to_id[token] = start_idx
                start_idx += 1

        self.id_to_char = {i:c for c, i in self.char_to_id.items()}
        self.pad_token_id = self.char_to_id["<PAD>"]
        self.unk_token_id = self.char_to_id["<UNK>"]

    def encode(self, text):
        preprocessed = re.split(r'(--|[,.:;?_!"()«»\'—\-\s])', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.char_to_id.get(s, self.unk_token_id) for s in preprocessed]
        return ids

    def decode(self, ids):
        ids = [id for id in ids if id != self.unk_token_id]
        text = " ".join([self.id_to_char[i] for i in ids])
        text = re.sub(r'\s+([,.:;_?!")\'»])', r'\1', text)
        text = re.sub(r'(["(\'_«])\s+', r'\1', text)
        return text
