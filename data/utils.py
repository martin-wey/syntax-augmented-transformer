import io
import re
import tokenize

from tree_sitter import Language

LANGUAGES = (
    'python',
    'java',
    'ruby',
    'javascript',
    'go',
    'php'
)

LANGUAGE_GRAMMARS = {
    'python': Language('grammars/languages.so', 'python'),
    'javascript': Language('grammars/languages.so', 'javascript'),
    'go': Language('grammars/languages.so', 'go'),
}


def remove_comments_and_docstrings_java_js(string):
    """Source: https://stackoverflow.com/questions/2319019/using-regex-to-remove-comments-from-source-files"""
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(_replacer, string)


# https://gist.github.com/maxpv/5ff921c1c721d91f96f2ea3883cef518
def remove_comments_php(s):
    for x in re.findall(r'("[^\n]*"(?!\\))|(//[^\n]*$|/(?!\\)\*[\s\S]*?\*(?!\\)/)', s, 8): s = s.replace(x[1], '')
    s = re.sub(r'(?m) *#.*\n?', '', s)
    return s


def remove_comments_and_docstrings_python(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out


def preprocess_code(code, lang):
    if lang == 'python':
        return remove_comments_and_docstrings_python(code)
    elif lang == 'javascript' or lang == 'go':
        return remove_comments_and_docstrings_java_js(code)
    elif lang == 'php':
        return remove_comments_php(code)


def get_u_subword(u, mapping):
    new_u = []
    for j, l in enumerate(u):
        new_u.append(l)
        new_u = new_u + ([1] * (len(mapping[j]) - 1))
    return new_u


def get_d_subword(d, mapping):
    new_x = []
    for j, l in enumerate(d):
        new_x = new_x + ([0] * (len(mapping[j]) - 1))
        new_x.append(l)
    return new_x


def get_c_subword(c, mapping):
    new_x = []
    for j, l in enumerate(c):
        new_x = new_x + ([1] * (len(mapping[j]) - 1))
        new_x.append(l)
    return new_x
