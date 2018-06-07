# Pickles the output of the model into the individual folders
import logging
import torch

from tqdm import tqdm

try:
   import cPickle as pickle
except:
   import pickle


def pickle_encoding(model, dataloader, use_cuda=False):
	# set model to eval mode
	model.eval()

	# loop through data batches
	for batch_idx, (X, video_path) in enumerate(tqdm(dataloader)):
		# Utilize GPU if enabled
		X = X.squeeze()
		video_path = video_path[0]

		if use_cuda:
			X = X.cuda()

		# Get outputs
		encoding = model(X)
		obj_to_save = torch.t(encoding)

		if use_cuda:
			obj_to_save = obj_to_save.cpu()

		logging.debug('Saving pickle to {}'.format(video_path))
		pickle.dump(obj_to_save, open(video_path, 'wb'))
		
		del obj_to_save, X, video_path