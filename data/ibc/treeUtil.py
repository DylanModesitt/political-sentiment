from itertools import *

# for iteration through tree
def isingle(obj):
    yield obj

class leafObj:

    def __init__(self, word, pos):
        self.word = word
        self.pos = pos
        self.finished = 1
        self.alpha = 0.0
        self.c1 = None
        self.c2 = None

    def __iter__(self):
        return isingle(self)

    def set_parent(self, par):
        self.parent = par

    def print_leaf(self):
        print(self.word, ': ', self.pos)

    def set_label(self, label):
        self.label = label


class nodeObj:

    def __init__(self, c1, c2, pos):
        self.c1 = c1
        self.c2 = c2
        self.pos = pos
        self.finished = 1
        self.alpha = 0.0

    def __iter__(self):
        return chain(isingle(self), iter(self.c1), iter(self.c2))

    def set_parent(self, par):
        self.parent = par

    def set_label(self, label):
        self.label = label

    def set_all_labels(self, label):
        for node in self:
            node.set_label(label)

    def set_all_betas(self, beta):
        for node in self:
            node.beta = beta

    def reset_finished(self):
        for node in self:
            node.finished = 0

    def get_words(self):
        return ' '.join([leaf.word for leaf in self.get_leaves()])

    def get_indices(self):
        return [leaf.leaf_index for leaf in self.get_leaves()]

    def get_leaves(self):
        leaves = []
        kids = [self.c1, self.c2]

        while kids:
            k = kids.pop(0)
            if isinstance(k, leafObj):
                leaves.append(k)

            else:
                kids.insert(0, k.c1)
                kids.insert(1, k.c2)

        return leaves

    def set_vectors(self, vecs):
        for index, leaf in enumerate(self.get_leaves()):
            leaf.vec = vecs[index]
