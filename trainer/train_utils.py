import logging
import torch
from tqdm import tqdm

from utils.metrics import accuracy, AverageMeter


def train_model(model, dataloader, loss_fn, optimizer, epoch, is_lstm, use_cuda=False, verbose=False):
	# set model to train mode
	model.train()
	top1 = AverageMeter()
	total_loss = 0

	# loop through data batches
	count = 0
	for batch_idx, (X, y) in enumerate(tqdm(dataloader)):
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
		# Compute loss
		predictions = model(X)
		
		count += predictions.shape[0]
		loss = loss_fn(predictions, y)
		total_loss += loss.item()
		
		# Compute running accuracy
		acc1 = accuracy(predictions.data, y, (1,))
		top1.update(acc1[0], batch_size)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if verbose:
			print('Progress [{0}/{1} ({2:.0f}%)]\tLoss:{3}'.format(count, len(dataloader.dataset), 
				100. * batch_idx / len(dataloader), loss.item()))

	total_loss /= count
	train_acc = top1.avg
	logging.info('Train Epoch: {} \tLoss: {:.6f} \t Training Acc: {:.2f}'.format(epoch, total_loss, train_acc))

	return total_loss, train_acc

def validate_model(model, dataloader, loss_fn, is_lstm, predictions_saver=None, use_cuda=False):
	# set the model to evaluation mode
	model.eval()
	top1 = AverageMeter()
	total_loss = 0

	count = 0
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
		count += predictions.shape[0]
		loss = loss_fn(predictions, y)

		if is_lstm and predictions_saver:
			prediction_indices = torch.argmax(predictions, dim=1)
			for batch_index in range(batch_size):
				predictions_saver.update([
					X['video_dirs'][batch_index].strip('/').split('/')[-1], # id
					y.numpy()[batch_index],					# label	
					prediction_indices.numpy()[batch_index]	# prediction
				])

		# measure accuracy
		acc1 = accuracy(predictions.data, y, (1,))
		top1.update(acc1[0], batch_size)
	
	return top1.avg
