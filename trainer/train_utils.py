import logging
import torch

def train_model(model, dataloader, epochs, loss_fn, optimizer, epoch, use_cuda):
	# set model to train mode
	model.train()

	# loop through data batches
	for batch_idx, (X, y) in enumerate(dataloader):
		# Utilize GPU if enabled
		if use_cuda:
			X['X'] = X['X'].cuda()
			y = y.cuda(async=True)

		# Compute and print loss
		predictions = model(X)
		loss = loss_fn(predictions, y)
		logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(X), len(dataloader.dataset),
				100. * batch_idx / len(dataloader), loss.item()))

		# Update weights after the dummy model is differentiable. Otherwise,
		# this will crash because there is nothing for the model to update.
		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()