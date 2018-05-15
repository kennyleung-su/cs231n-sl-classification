import logging
import torch

def train_model(model, dataloader, epochs, loss_fn, optimizer, epoch, use_cuda=False):
	# set model to train mode
	model.train()

	# loop through data batches
	for batch_idx, (X, y) in enumerate(dataloader):
		# Utilize GPU if enabled
		if use_cuda:
			X['X'] = X['X'].cuda()
			y = y.cuda(async=True)

		# Compute loss and accuracy
		predictions = model(X)
		loss = loss_fn(predictions, y)
		top1 = AverageMeter()
		acc1 = accuracy(predictions.data, y, (1,))
		top1.update(acc1[0], X['X'].size(0))

		logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Train Acc: {:.2f}'.format(epoch, batch_idx * len(X), len(dataloader.dataset),
				100. * batch_idx / len(dataloader), loss.item(), top1.avg))

		# Update weights after the dummy model is differentiable. Otherwise,
		# this will crash because there is nothing for the model to update.
		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def validate_model(model, dataloader, loss_fn, use_cuda=False):
	# set the model to evaluation mode
	model.eval()
	top1 = AverageMeter()

	for i, (X, y) in enumerate(dataloader):
		if use_cuda:
			X['X'] = X['X'].cuda()
			y = y.cuda(async=True)
		# compute output
		predictions = model(X)

		# measure accuracy
		acc1 = accuracy(predictions.data, y, (1,))
		top1.update(acc1[0], X['X'].size(0))

	return top1.avg

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
