# Generating Text with Recurrent Neural Networks

Source: https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf
Authors: Ilya Sutskever, James Martens, Geoffrey Hinton

This project implements a recurrent neural network that generates text.

## Examples

The generations don't make a lot of sense. They could definitely be further optimized, but they do come up with some fun results.

Robert Frost
```
The mad for that some begoging to reath the or the seem
At earth so blast both had to strance
One me and the could to say see.
The vogess to strimes to her have as man bou
```

Ernest Hemingway
```
"He he the fish," he was.

He rose with the fish and to see the skiff and the bait.  He waPero the sure the line and the great the working come.

"I will shave it the boy said alm stead.  "But with the boy to the water the old man said.  "But I come the fish."
```

## Requirements + Versions

- [Python 3.6](https://www.python.org/)
- [PyTorch 1.0](https://pytorch.org/)
- [Unidecode 1.0](https://pypi.org/project/Unidecode)

## Running

All the config is done within the main file. Tweak the variables from within.

```
$ python3 src/main.py
```

Results will be saved in `/output`