"""Utils for language models."""

import re
import numpy as np
import json
from collections import Counter

# ------------------------
# torchtext-free tokenizer / vocab (torchtext is incompatible with torch 2.11).
# basic_english_tokenizer reproduces torchtext's 'basic_english' normalization;
# build_vocab is a deterministic frequency vocab with the same special tokens.

_BASIC_PATTERNS = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)",
                   r"\!", r"\?", r"\;", r"\:", r"\s+"]
_BASIC_REPLACE = [" '  ", "", " . ", " ", " , ", " ( ", " ) ",
                  " ! ", " ? ", " ", " ", " "]
_BASIC_COMPILED = [(re.compile(p), r) for p, r in zip(_BASIC_PATTERNS, _BASIC_REPLACE)]

SPECIALS = ["<pad>", "<cls>", "<unk>", "<eos>"]  # ids 0,1,2,3


def basic_english_tokenizer(line):
    """Lowercase, pad punctuation, split on whitespace (torchtext basic_english)."""
    line = line.lower()
    for pat, repl in _BASIC_COMPILED:
        line = pat.sub(repl, line)
    return line.split()


def build_vocab(token_lists, max_tokens=32000, specials=SPECIALS):
    """Deterministic vocab from token lists: specials first (ids 0..3), then most
    frequent tokens (ties broken alphabetically) up to max_tokens. Returns
    (itos, stoi)."""
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)
    itos = list(specials)
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    for tok, _ in ordered:
        if max_tokens and len(itos) >= max_tokens:
            break
        itos.append(tok)
    stoi = {t: i for i, t in enumerate(itos)}
    return itos, stoi


def encode_tokens(tokens, stoi, max_len):
    """[<cls>] + token ids (unknown -> <unk>), padded with <pad>(0)/truncated to
    max_len. Returns (ids, true_length_capped_at_max_len)."""
    ids = [stoi["<cls>"]] + [stoi.get(t, stoi["<unk>"]) for t in tokens]
    true_len = min(len(ids), max_len)
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids, true_len


# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def letter_to_index(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return index

def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string

    Return:
        indices: int list with length len(word)
    '''
    print('num of letters (classes):', NUM_LETTERS)
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words

    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words

    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    """Converts target into one-hot.

    Args:
        size: Size of vector.
        val: Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec

def tokenizer(text, max_len, max_tokens=32000):
    """Backward-compatible: build vocab from all `text` and encode. Prefer
    basic_english_tokenizer + build_vocab + encode_tokens directly when a
    train-only vocab is required."""
    tokenized = [basic_english_tokenizer(t) for t in text]
    itos, stoi = build_vocab(tokenized, max_tokens=max_tokens)
    text_list = [encode_tokens(toks, stoi, max_len)[0] for toks in tokenized]
    return stoi, text_list
