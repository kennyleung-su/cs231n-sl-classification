import logging
import torch


def train_model(model, dataloader, epochs, loss_fn, optimizer, epoch):
	"""TODO: Implement model parameter updates."""

	# TODO: Iterate over the data in batches, perhaps taking in a dataloader
	# object instead.
	for batch_idx, (X, y) in enumerate(dataloader):
		# Compute and print loss
		predictions = model(X)
		loss = loss_fn(predictions, y)
		logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(X), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

		# TODO: Update weights after the dummy model is differentiable. Otherwise,
		# this will crash because there is nothing for the model to update.
		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# TODO: Run on the validation set, with frequency depending on
		# on the configuration.

		# TODO: Pickle the best seen model. Use model._best_accuracy
		# (change to model.best_val_accuracy?) to determine if this is the case.
