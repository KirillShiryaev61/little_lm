"""Реализация простого токенайзера по словам и знакам препинания."""

import re

class SimpleTokenizerV1():

    def __init__(self, vocab):
        self.char_to_id = vocab

        self.special_tokens = ["<PAD>", "<UNK>"]
        for idx, token in enumerate(self.special_tokens, start=len(self.char_to_id)):
            self.char_to_id[token] = idx

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
        text = " ".join([self.id_to_char[i] for i in ids])
        text = re.sub(r'\s+([,.:;_?!")\'»])', r'\1', text)
        text = re.sub(r'(["(\'_«])\s+', r'\1', text)
        return text
