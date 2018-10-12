import utils
import string

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers):
		super(RNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers

		self.encoder = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
		self.decoder = nn.Linear(hidden_size, output_size)

	def forward(self, input, hidden):
		input = self.encoder(input.view(1, -1))
		output, hidden = self.gru(input.view(1, 1, -1), hidden)
		output = self.decoder(output.view(1, -1))
		return output, hidden

	def init_hidden(self):
		return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))

# Feed one char at a time to network
# Use outputs as a probability distribution
def generate(network, prime, predict_length, temperature):
	hidden = network.init_hidden()
	prime_input = utils.char_tensor(prime)
	predicted = prime

	# Use prime string to build hidden state
	for p in range(len(prime)):
		_, hidden = network(prime_input[p], hidden)

	input_char = prime_input[-1]

	for p in range(predict_length):
		output, hidden = network(input_char, hidden)

		# Sample from the network using the distribution
		output_distribution = output.data.view(-1).div(temperature).exp()
		top_index = torch.multinomial(output_distribution, 1)[0]

		# Add predicted char to string and use as next input
		predicted_char = characters[top_index]
		predicted += predicted_char
		input_char = utils.char_tensor(predicted_char)

	return predicted

def train(network, input_tensor, target_tensor, chunk_length, optimizer, criterion):
	hidden = network.init_hidden()
	network.zero_grad()
	loss = 0

	for c in range(chunk_length):
		output, hidden = network(input_tensor[c], hidden)
		loss += criterion(output, target_tensor[c].unsqueeze(0))

	loss.backward()
	optimizer.step()

	return loss.data.item() / chunk_length