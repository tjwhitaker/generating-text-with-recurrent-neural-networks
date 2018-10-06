import random
import string

import torch
from torch.autograd import Variable

characters = string.printable

# Take random group of x characters from file
def random_chunk(file, chunk_length):
	start = random.randint(0, len(file) - chunk_length)
	end = start + chunk_length + 1
	return file[start:end]

# Convert chunk of characters into a long tensor
def char_tensor(chunk):
	tensor = torch.zeros(len(chunk)).long()
	for c in range(len(chunk)):
		tensor[c] = characters.index(chunk[c])

	return Variable(tensor)

# Pair an input with a target according to chunk
# Ex: chunk: 'abc', input: 'ab', target: 'bc'
def random_training_set(file, chunk_length):
	chunk = random_chunk(file, chunk_length)
	input_tensor = char_tensor(chunk[:-1])
	target_tensor = char_tensor(chunk[1:])
	return input_tensor, target_tensor