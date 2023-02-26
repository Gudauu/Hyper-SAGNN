import os
import warnings
import torch
import torch.nn.functional as F

# Load PyTorch custom op library
word2vec = torch.ops.load_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
warnings.filterwarnings("ignore")

class Word2Vec_Skipgram_Data(object):
    """Word2Vec model (Skipgram)."""

    def __init__(
            self,
            train_data,
            num_samples,
            batch_size,
            window_size,
            min_count,
            subsample,
            device):
        self.train_data = train_data
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.window_size = window_size
        self.min_count = min_count
        self.subsample = subsample
        self.device = device
        self._word2id = {}
        self._id2word = []
        self.build_graph()

    def build_graph(self):
        """Build the graph for the full model."""
        # The training data. A text file.
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
         labels) = word2vec.skipgram_word2vec(filename=self.train_data,
                                              batch_size=self.batch_size,
                                              window_size=self.window_size,
                                              min_count=self.min_count,
                                              subsample=self.subsample)

        (self.vocab_words, self.vocab_counts,
         self.words_per_epoch) = (words.cpu().numpy().tolist(),
                                  counts.cpu().numpy().tolist(),
                                  words_per_epoch.item())

        self.vocab_size = len(self.vocab_words)
        print("Data file: ", self.train_data)
        print("Vocab size: ", self.vocab_size - 1, " + UNK")
        print("Words per epoch: ", self.words_per_epoch)
        self._examples = examples.to(self.device)
        self._labels = labels.to(self.device)
        self._id2word = self.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        id2word = []
        for i, w in enumerate(self._id2word):
            try:
                id2word.append(int(w))
            except BaseException:
                id2word.append(w)

        self._id2word = id2word

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = self._labels.view(self.batch_size, 1)

        # Negative sampling.
        self.sampled_ids, _, _ = (torch.ops.torch_word2vec_ops.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,
            range_max=self.vocab_size,
            distortion=0.75,
            unigrams=torch.tensor(self.vocab_counts, dtype=torch.float32).to(self.device)))

    def next_batch(self):
        """Train the model."""

        initial_epoch, e, l, s, words = (self._epoch.item(),
                                         self._examples.cpu().numpy().tolist(),
                                         self._labels.cpu().numpy().tolist(),
                                         self.sampled_ids.cpu().numpy().tolist(),
                                         self._words)

        # All + 1 because of the padding_idx
        e_new = []