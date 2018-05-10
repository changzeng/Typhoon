# encoding: utf-8

# author: liaochangzeng
# github: https://github.com/changzeng

import numpy as np
from tensorflow.contrib import learn

class Vocabulary(learn.preprocessing.VocabularyProcessor):
    def __init__(self, max_document_length, min_frequency=0, vocabulary=None):
        def tokenizer_fn(iterator):
            for sen in iterator:
                yield sen.split(" ")
        self.sup = super(Vocabulary, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
