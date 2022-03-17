# syntax-augmented-transformer

You can download and preprocess data using the following command.
```shell
unzip dataset.zip
cd dataset
bash run.sh 
cd ..
```

Install tree-sitter grammars:

```sh
mkdir grammars
cd grammars
git clone https://github.com/tree-sitter/tree-sitter-python.git
git clone https://github.com/tree-sitter/tree-sitter-javascript.git
git clone https://github.com/tree-sitter/tree-sitter-go.git
```

Build grammars:

```sh
python data/build_grammars.py
```
