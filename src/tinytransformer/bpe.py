import json
from collections import Counter

BASE_VOCAB=256

def get_pairs(word):
    return set(zip(word, word[1:]))

def merge_pair(word, pair, new_id):
    out =[]
    i = 0
    while i < len(word):
        if i+1<len(word) and (word[i], word[i+1])==pair:
            out.append(new_id)
            i+=2
        else:
            out.append(word[i])
            i+=1
    
    return out

class BPE:
    def __init__(self, merges=()):
        self.merges = list(merges)
        self.rank = {pair: i for i,pair in enumerate(self.merges)}
        self.vocab = {i : bytes([i]) for i in range(BASE_VOCAB)}
        for i, (a,b) in enumerate(self.merges):
            self.vocab[BASE_VOCAB+i] = self.vocab[a]+self.vocab[b]

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size):

        words = Counter(" " + w for w in text.split())
        words = [[list(w.encode("utf-8")), n] for w,n in words.items()]

        merges = []
        while BASE_VOCAB+len(merges)<vocab_size:
            counts= Counter()
            for ids, n in words:
                for pair in zip(ids,ids[1:]):
                    counts[pair]+=n

            if not counts:
                break

            pair, freq = counts.most_common(1)[0]
            if freq <2:
                break

            new_id = BASE_VOCAB + len(merges)

            merges.append(pair)

            if len(merges) % 500 == 0:
                print(f"  merge {len(merges):5d}  freq {freq:6d}")

            words = [[merge_pair(ids, pair, new_id), n] for ids,n in words]

        return cls(merges)

    def encode(self, text):
        ids = []

        for w in text.split():
            word = list((" " + w).encode("utf-8"))

            while len(word)>1:
                best = None
                best_rank = None
                for pair in get_pairs(word):
                    rank = self.rank.get(pair)
                    if rank is not None and (best_rank is None or rank<best_rank):
                        best=pair
                        best_rank=rank

                if best is None:
                    break
                word = merge_pair(word, best, BASE_VOCAB+best_rank)
            ids.extend(word)
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace").lstrip(" ")

    def save(self,path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges":[list(p) for p in self.merges]}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls([tuple(p) for p in data["merges"]])

    
def run(paths, vocab_size=6000, out="data/bpe.json"):
    lines = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            lines.extend(f.read().splitlines()[:10000])

    bpe = BPE.train("\n".join(lines), vocab_size)
    bpe.save(out)

    sample = lines[0]
    ids = bpe.encode(sample)
    assert bpe.decode(ids) == sample
    print(len(bpe.merges), "merges,", len(ids), "tokens for:", sample)

    # did the merges actually compress held-out text?
    for path in paths:
        probe = " ".join(open(path, encoding="utf-8").read().splitlines()[10000:11000])
        n_ids = len(bpe.encode(probe))
        if not n_ids:
            continue
        print(f"{path}: {len(probe.encode('utf-8')) / n_ids:.2f} bytes/token")

    # early merges should look like frequent text, late ones like long chunks
    def piece(i):
        return bpe.vocab[BASE_VOCAB + i].decode("utf-8", "replace")

    print("first merges:", [piece(i) for i in range(10)])
    print("last merges: ", [piece(i) for i in range(len(bpe.merges) - 10, len(bpe.merges))])