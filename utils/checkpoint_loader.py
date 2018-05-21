import os
import glob
import re

# return the path to the checkpoint with the best accuracy if found, else return none
def get_best_checkpoint(config):
	best_checkpoint_file = None
	best_acc = -1

	if os.path.exists(config.checkpoint_path):
		for checkpoint in glob.glob(os.path.join(config.checkpoint_path, '*-best-*.pkl')):
			acc = int(re.search(r'-best-([0-9]+).pkl', checkpoint).group(1))
			if acc > best_acc:
				best_checkpoint_file = checkpoint
				best_acc = acc

	return best_checkpoint_file    