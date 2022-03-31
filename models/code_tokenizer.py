from typing import List, Dict

from tokenizers import NormalizedString, PreTokenizedString, Tokenizer, pre_tokenizers

from data.code2ast import code2ast, get_token


def map_byte_to_codepoint_offset(text: str) -> Dict[int, int]:
    mapping = {}
    byte_offset = 0
    for codepoint_offset, character in enumerate(text):
        mapping[byte_offset] = codepoint_offset
        byte_offset += len(character.encode('utf8'))
    return mapping


def tokenize_code(code, parser, lang):
    g, code_pre = code2ast(code, parser, lang)
    correspondence = map_byte_to_codepoint_offset(code_pre)
    return [(get_token(code_pre, g.nodes[t]['start'], g.nodes[t]['end']),
             correspondence[g.nodes[t]['start']],
             correspondence[g.nodes[t]['end']] if g.nodes[t]['end'] in correspondence else len(code_pre))
            for t in sorted([n for n in g if g.nodes[n]['is_terminal']], key=lambda n: g.nodes[n]['start'])]


class CodePreTokenizer:
    def __init__(self, parser, lang):
        self.parser = parser
        self.lang = lang

    def code_split(self, i: int,
                   normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        for token, start, stop in tokenize_code(str(normalized_string), self.parser, self.lang):
            splits.append(normalized_string[start:stop])
        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.code_split)


def save_code_tokenizer(tokenizer, path):
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.save(path)


def load_code_tokenizer(path, parser, lang):
    loaded_tokenizer = Tokenizer.from_file(path)
    loaded_tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(CodePreTokenizer(parser, lang=lang))
    return loaded_tokenizer
