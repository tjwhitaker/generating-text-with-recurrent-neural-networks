import torch
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers=1):
		super(RNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers

		self.encoder = nn.Embedding(self.input_size, self.hidden_size)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers)
		self.decoder = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden):
		input = self.encoder(input.view(1, -1))
		output, hidden = self.gru(input.view(1, 1, -1), hidden)
		output = self.decoder(output.view(1, -1))
		return output, hidden

	def init_hidden(self):
		return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))