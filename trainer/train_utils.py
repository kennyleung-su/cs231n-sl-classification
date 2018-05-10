import torch


def train_model(X, y,model, epochs, loss_fn, optimizer, lr):
	"""TODO: Implement model parameter updates."""

	predictions = model(X)

	# Compute and print loss
	loss = loss_fn(predictions, y)
	logging.info('Loss:', loss.item())

	# Zero gradients, perform a backward pass, and update the weights.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
