"""Adapted from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7."""

import numpy as np
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
    if padded_length == tensor.size(dim):
        return tensor
    assert pad_size[dim] > 0, 'This tensor exceeds the possible padding size.'
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
            batch - a list of dicts with keys 'frames', 'label', 'seq_len'

        Returns:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        frames_and_labels = list(map(lambda x: (pad_tensor(
                                x['frames'], padded_length=self.max_seq_len,
                                dim=self.dim), x['label']), batch))

        # Find an ordering of the sequences; LSTM requires that rows are
        # given in descending order of time sequence length.
        seq_lens = [x['seq_len'] for x in batch]
        seq_lens_indices = np.argsort(seq_lens)[::-1]

        # Stack frames in batches. The result is a (N, T, H, W, C) tensor.
        frames = [x[0] for x in frames_and_labels]
        frames = [frames[i] for i in seq_lens_indices]
        frames = torch.stack(frames, dim=0)

        # Additional metadata.
        # video_dirs = np.array([x['video_dir'] for x in batch])
        # video_dirs = video_dirs[seq_lens_indices]

        # Rearrange all elements in order of decreasing sequence length
        seq_lens = np.sort(seq_lens)[::-1]

        labels = [x[1] for x in frames_and_labels]
        labels = [labels[i] for i in seq_lens_indices]
        labels = torch.LongTensor(labels)
        seq_lens = torch.from_numpy(np.stack(seq_lens))

        return {'X': frames, 'seq_lens': seq_lens}, labels

    def __call__(self, batch):
        return self.pad_collate(batch)
