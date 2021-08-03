from torch.utils.data.sampler import Sampler
import torch

class CountSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, batch_size, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.batch_size = batch_size
        self.video_lengths = data_source.video_lengths

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples


    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        while num_batches > 0:
            sampled = []
            while len(sampled) < self.batch_size:

                if len(sampled) == 0:
                    sampled.append(torch.argmax(scores))
                else:
                    sampled.append(torch.argmax(scores))
            yield sampled
            num_batches -= 1

    def __len__(self):
        return self.num_samples
