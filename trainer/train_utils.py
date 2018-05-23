import logging
import torch

def train_model(model, dataloader, loss_fn, optimizer, epoch, is_lstm, use_cuda=False, verbose=False):
	# set model to train mode
	model.train()
	total_loss = 0

	# loop through data batches
	count = 0
	for batch_idx, (X, y) in enumerate(dataloader):
		batch_size = -1
		# Utilize GPU if enabled
		if use_cuda:
			if is_lstm:
				X['X'] = X['X'].cuda()
			else:
				X = X.cuda()
			y = y.cuda(async=True)

		if is_lstm:
			batch_size = X['X'].size(0)
		else:
			batch_size = X.size(0)
		# Compute loss and accuracy
		predictions = model(X)
		count += predictions.shape[0]
		loss = loss_fn(predictions, y)
		total_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if verbose:
			print('Progress [{}/{} ({:.0f}%)]'.format(count, len(dataloader.dataset), 
				100. * batch_idx / len(dataloader)))

	total_loss /= count
	logging.info('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))

def validate_model(model, dataloader, loss_fn, is_lstm, use_cuda=False):
	# set the model to evaluation mode
	model.eval()
	top1 = AverageMeter()
	total_loss = 0

	for i, (X, y) in enumerate(dataloader):
		batch_size = -1
		# Utilize GPU if enabled
		if use_cuda:
			if is_lstm:
				X['X'] = X['X'].cuda()
			else:
				X = X.cuda()
			y = y.cuda(async=True)

		if is_lstm:
			batch_size = X['X'].size(0)
		else:
			batch_size = X.size(0)
		# compute output
		predictions = model(X)
		loss = loss_fn(predictions, y)
		total_loss += loss.item()

		# measure accuracy
		acc1 = accuracy(predictions.data, y, (1,))
		top1.update(acc1[0], batch_size)

	return top1.avg, total_loss

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
