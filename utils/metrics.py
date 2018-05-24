"""Utilities for computing and saving model metrics."""

import csv
import logging
import matplotlib.pyplot as plt

def accuracy(output, target, topk=(1,)):
	# specifies the the precision of the top k values
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class MetricsCsvSaver(object):
	"""Stores metrics and writes them to the output path."""
	def __init__(self, output_path, header):
		self._output_path = output_path
		self._header = header
		self._rows = []

	def __del__(self):
		"""Saves the file, so all data is saved even if an exception is thrown."""
		logging.info('Saving metrics file to: {0}'.format(self._output_path))
		with open(self._output_path, 'w') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(self._header)
			writer.writerows(self._rows)

	def update(self, vals):
		self._rows.append(vals)

	def plot(self, plot_output_path, title, xlabel, ylabel):
		"""Naively plots the first col as the X-axis, vs. the second col as the Y-axis."""
		logging.info('Saving {0} vs. {1} plot to: {2}'.format(ylabel, xlabel, plot_output_path))
		X, Y = [list(a) for a in (zip(*self._rows))]
		plt.figure()
		plt.plot(X, Y)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.savefig(plot_output_path)


# Light wrapper classes around common saver instances.

class AccuracySaver(MetricsCsvSaver):
	def __init__(self, output_path):
		super(AccuracySaver, self).__init__(output_path=output_path,
											header=["epoch", "accuracy"])


class LossSaver(MetricsCsvSaver):
	def __init__(self, output_path):
		super(LossSaver, self).__init__(output_path=output_path,
											header=["epoch", "loss"])


class PredictionSaver(MetricsCsvSaver):
	def __init__(self, output_path):
		super(PredictionSaver, self).__init__(output_path,
											header=["id", "label", "pred"])

	def plot(self):
		raise NotImplemented("Confusion matrix plotting not yet implemented.")
