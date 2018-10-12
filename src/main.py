import model
import settings
import utils

import random
import string
import unidecode

import torch
import torch.nn as nn

# Config
settings.init()

input_string = unidecode.unidecode(open(INPUT_PATH, 'r').read())

network = model.RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
	input_tensor, target_tensor = utils.random_training_set(input_string, CHUNK_LENGTH)

	loss = model.train(network, input_tensor, target_tensor, CHUNK_LENGTH, optimizer, criterion)

	if epoch % PRINT_EVERY == 0:
		print(loss)
		print(model.generate(network, random.choice(string.ascii_uppercase), PREDICT_LENGTH, TEMPERATURE), '\n')

with open(OUTPUT_PATH, 'a') as file:
	for i in range(10):
		file.write(model.generate(network, random.choice(string.ascii_uppercase), 200, TEMPERATURE))