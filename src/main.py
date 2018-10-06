import model
import utils

import random
import string
import unidecode

import torch
import torch.nn as nn


# Config
epochs = 20000
print_every = 100
learning_rate = 0.005
predict_length = 100
temperature = 0.6
chunk_length = 200

characters = string.printable

input_string = unidecode.unidecode(open('../input/notorious-tupac.txt').read())

network = model.RNN()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
	input_tensor, target_tensor = utils.random_training_set(input_string, chunk_length)

	loss = model.train(network, input_tensor, target_tensor, chunk_length, optimizer, criterion)

	if epoch % print_every == 0:
		print(loss)
		print(model.generate(network, random.choice(string.ascii_uppercase), predict_length, temperature), '\n')

song = model.generate(network, random.choice(string.ascii_uppercase), 2000, temperature)

with open('../output/roborap.txt', w) as file:
	file.write(song)