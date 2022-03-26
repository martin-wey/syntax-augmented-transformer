import os
import unittest

from tree_sitter import Language, Parser
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import models, Tokenizer, trainers, pre_tokenizers

from models.code_tokenizer import CodePreTokenizer
from data.code2ast import code2ast

code_python = """def maximum(a, b):
    if a > b:
        return a
    return b"""

code_python_string = """def count_ocurrences(input="1-2-3-4"):
    splits = input.split("-")
    return len(splits)"""


class TestCodeTokenizer(unittest.TestCase):
    @staticmethod
    def get_python_setup():
        py_lang = Language('grammars/languages.so', 'python')
        parser = Parser()
        parser.set_language(py_lang)
        pretokenizer = PreTokenizer.custom(CodePreTokenizer(parser, lang='python'))
        return parser, pretokenizer

    def test_python_pretokenizer(self):
        parser, pretokenizer = TestCodeTokenizer.get_python_setup()

        _, pre_code = code2ast(code_python, parser, lang='python')
        tokens = pretokenizer.pre_tokenize_str(code_python)
        for t, (a, b) in tokens:
            self.assertEqual(t, pre_code[a:b])

        _, pre_code_string = code2ast(code_python_string, parser, lang='python')
        tokens_string = pretokenizer.pre_tokenize_str(code_python_string)
        print(tokens_string)
        for t, (a, b) in tokens_string:
            self.assertEqual(t, pre_code_string[a:b])

    def test_wordpiece_code(self):
        parser, pretokenizer = TestCodeTokenizer.get_python_setup()

        tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))
        tokenizer.pre_tokenizer = pretokenizer
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=100, special_tokens=special_tokens)
        tokenizer.train_from_iterator([code_python, code_python_string], trainer=trainer)

        encoding_1 = tokenizer.encode(code_python, code_python_string)
        toks_1 = encoding_1.tokens

        #save wordpiece, we replace the pretok beacuse the custom one is not serializable
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.save("tokenizer.json")
        loaded_tokenizer = Tokenizer.from_file("tokenizer.json")

        loaded_tokenizer.pre_tokenizer = pretokenizer
        encoding_2 = tokenizer.encode(code_python, code_python_string)
        toks_2 = encoding_2.tokens
        self.assertTrue(toks_1, toks_2)

        os.remove("tokenizer.json")


if __name__ == '__main__':
    unittest.main()
