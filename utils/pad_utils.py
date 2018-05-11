"""Adapted from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7."""

import torch


def pad_tensor(tensor, padded_length, dim):
    """
    Args:
        tensor: torch.Tensor
        padded_length: int, length after padding
        dim: int, dimension to pad

    Returns:
        torch.Tensor where dim is zero-padded to padded_length
    """
    pad_size = list(tensor.shape)
    pad_size[dim] = padded_length - tensor.size(dim)
    assert pad_size[dim] >= 0, 'This tensor exceeds the possible padding size.'
    return torch.cat([tensor, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    A variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, max_seq_len, dim=0):
        """
        Args:
            max_seq_len: int, the length to pad the input out to
            dim: int, the dimension to be padded (dimension of time in sequences)
        """
        self.max_seq_len = max_seq_len
        self.dim = dim

    def pad_collate(self, batch):
        """
        Args:
            batch - a list of dicts with keys 'frames', 'label'

        Returns:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        frames_and_labels = map(lambda x: (pad_tensor(
                                x['frames'], padded_length=self.max_seq_len,
                                dim=self.dim), x['label']), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], frames_and_labels), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], frames_and_labels))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)