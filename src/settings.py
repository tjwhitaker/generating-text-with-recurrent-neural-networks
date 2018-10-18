import string

EPOCHS = 2000
PRINT_EVERY = 100
LEARNING_RATE = 0.005
PREDICT_LENGTH = 100
TEMPERATURE = 0.5
CHUNK_LENGTH = 200
CHARACTERS = string.printable
INPUT_PATH = '../input/old-man-and-the-sea.txt'
OUTPUT_PATH = '../output/old-man-and-the-sea.txt'
INPUT_SIZE = OUTPUT_SIZE = len(CHARACTERS)
HIDDEN_SIZE = 100
NUM_LAYERS = 1