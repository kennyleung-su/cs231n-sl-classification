"""Utilities for computing and saving model metrics."""

import csv
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

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
	"""Computes and stores the average and current value, among other statistics."""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.min = float('inf')
		self.max = float('-inf')

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		# Update the range of visited values.
		if val < self.min:
			self.min = val
		if val > self.max:
			self.max = val


class MetricsCsvSaver(object):
	"""Stores metrics and writes them to the output path."""
	def __init__(self, output_path, header, aggregate_index=None):
		self._output_path = output_path
		self._header = header
		self._rows = []

		# Whether to use an AverageMeter to track values at a given index of update().
		self._aggregate_index = aggregate_index
		if self._aggregate_index:
			self.meter = AverageMeter()


	def __del__(self):
		"""Saves the file, so all data is saved even if an exception is thrown."""
		logging.info('Saving metrics file to: {0}'.format(self._output_path))
		with open(self._output_path, 'w') as f:
			writer = csv.writer(f, lineterminator='\n')
			writer.writerow(self._header)
			writer.writerows(self._rows)

	def update(self, vals):
		self._rows.append(vals)
		if self._aggregate_index:
			if not self.meter:
				raise ValueError('No AverageMeter has been initialized to aggregate metrics.')
			self.meter.update(vals[self._aggregate_index])

	def get_plot_points(self):
		return [ list(a) for a in (zip(*self._rows)) ]

	def plot(self, plot_output_path, title, xlabel, ylabel):
		"""Naively plots the first col as the X-axis, vs. the second col as the Y-axis."""
		logging.info('Saving {0} vs. {1} plot to: {2}'.format(ylabel, xlabel, plot_output_path))
		logging.info()
		X, Y = self.get_plot_points()
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
											header=["epoch", "accuracy"],
											aggregate_index=1)


class LossSaver(MetricsCsvSaver):
	def __init__(self, output_path):
		super(LossSaver, self).__init__(output_path=output_path,
										header=["epoch", "loss"],
										aggregate_index=1)


class PredictionSaver(MetricsCsvSaver):
	def __init__(self, output_path, num_classes):
		super(PredictionSaver, self).__init__(output_path,
											header=["id", "target", "pred"])
		self._num_classes = num_classes
		self._confusion_matrix = np.zeros((num_classes, num_classes))

	def update(self, vals):
		# Receives a triplet of (id, target, pred)
		id, target, pred = vals
		self._confusion_matrix[target, pred] += 1
		self._rows.append(vals)
		super(PredictionSaver, self).update

	def plot(self, plot_output_path, title):
		# TODO: Support having axis labels as the human-readable gesture names themselves.
		cm = pd.DataFrame(self._confusion_matrix, range(self._num_classes), range(self._num_classes))
		plt.figure()
		sn.set(font_scale=1.4)  #for label size
		sn.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap="Blues")
		plt.savefig(plot_output_path)
