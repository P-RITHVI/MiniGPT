from collections import Counter, defaultdict
import math

class BigramBaseline:
    """Add-one-smoothed character bigram baseline; BOS is a dedicated context."""
    def fit(self, sequences, vocab_size):
        self.vocab_size=vocab_size; self.counts=defaultdict(Counter)
        for seq in sequences:
            previous=vocab_size
            for token in seq:
                self.counts[previous][int(token)]+=1; previous=int(token)
        return self
    def loss(self,sequences):
        total=0; n=0
        for seq in sequences:
            previous=self.vocab_size
            for token in seq:
                count=self.counts[previous][int(token)]; denom=sum(self.counts[previous].values())+self.vocab_size
                total-=math.log((count+1)/denom); n+=1; previous=int(token)
        return total/n if n else float("nan")
