import random
import string
import torch
from torch.autograd import Variable
from settings import CHARACTERS

# Take random group of x characters from string 
def random_chunk(data, chunk_length):
	start = random.randint(0, len(data) - chunk_length)
	end = start + chunk_length + 1
	return data[start:end]

# Convert chunk of characters into a long tensor
def char_tensor(chunk):
	tensor = torch.zeros(len(chunk)).long()
	for c in range(len(chunk)):
		tensor[c] = CHARACTERS.index(chunk[c])

	return Variable(tensor)

# Pair an input with a target according to chunk
# Ex: chunk: 'abc', input: 'ab', target: 'bc'
def random_training_set(data, chunk_length):
	chunk = random_chunk(data, chunk_length)
	input_tensor = char_tensor(chunk[:-1])
	target_tensor = char_tensor(chunk[1:])
	return input_tensor, target_tensor