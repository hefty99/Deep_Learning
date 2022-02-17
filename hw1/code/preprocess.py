import gzip
import numpy as np


def get_data(inputs_file_path='/Users/loganheft/Downloads/data/train-images.gz', labels_file_path='/Users/loganheft/Downloads/data/train-labels.gz', num_examples=60000):

	image_size = 28
	normalization = 1/255.0
	offset16 = 16
	offset8 = 8
	image_size = 28
	
	with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
		file_content = bytestream.read()
		inputs_array = (1/255.0) * np.frombuffer(file_content,
						dtype=np.uint8, count=-1, offset=offset16)
		inputs_array = np.reshape(
			inputs_array, (num_examples, image_size * image_size))
	
	with open(labels_file_path, 'rb') as g, gzip.GzipFile(fileobj=g) as bytestream:
		file = bytestream.read()
		labels_arr = np.frombuffer(file, dtype=np.uint8, count=-1, offset=offset8)
	return (inputs_arr, labels_arr)
