# system
import pickle

# lib
import dill


def get_ibc_data(use_neutral=False,
                 use_subsampling=False,
                 return_subsampling_depths=False,
                 labels=(0, 1, 2)):
    """
    Samples the annotated IBC data. This dataset can be described
    more here. http://www.cs.umd.edu/~miyyer/ibc/.

    This dataset has sub-sentance labeled political sentiment
    data. Structured like (for example),

        (The affordable care act is great for americans, liberal)
                           |
                         |   |
                        |     |
                       |       |
                      N/A   (is great for americans, concervative)
                                     |
                                   |   |
                                 |       |
                                N/A   (for americans, neutral)

    You can get the fully annotated samples (with subsamples) by givinng
    use_subsampling as true.

    The data is returned as two lists. X is uncleaned text. Y is the label.
    If using subsampling, you can also choose to get the tree depth of the sample.
    Because the annotations run along a single path, there will only be one
    sample per depth per sentence.

    You can choose whhat labels you want by giving the labels parameter
    as a list or tuple of labels in the order of (liberal, conservative, neutral).
    The default is 0,1, or 2.

    :param use_neutral: whether or not to use neutral in the data
    :param use_subsampling: whether or not to subsample (give all parts of tree)
    :param return_subsampling_depths: whether or not to return the sampling depths
    :param labels:
    :return: X, Y, P?
    """

    lib, con, neutral = pickle.load(open('./data/ibc/ibcData.pkl', 'rb'))
    print(len(lib))
    print(len(con))

    if not use_neutral:
        neutral = []

    if not use_subsampling:

        extract = lambda x: [e.get_words() for e in x]
        lib, con, neutral = extract(lib), extract(con), extract(neutral)

        X = [*lib, *con, *neutral]
        Y = [*[labels[0]]*len(lib),
             *[labels[1]]*len(con),
             *[labels[2]]*len(neutral)]

        if return_subsampling_depths:
            raise ValueError('there is no subsampling positions if'
                             'this function is called without sub'
                             'sampling')

        return X, Y

    else:

        label_map = {
            'liberal': labels[0],
            'conservative': labels[1],
            'neutral': labels[2]
        }

        X, Y, P = [], [], []
        entries = [*lib, *con, *neutral]
        for tree in entries:
            depth = 0
            for node in tree:
                if hasattr(node, 'label'):
                    depth += 1
                    if use_neutral or node.label.lower() != 'neutral':
                        X.append(node.get_words())
                        Y.append(label_map[node.label.lower()])
                        P.append(depth)

        if return_subsampling_depths:
            return X, Y, P

        return X, Y


if __name__ == '__main__':

    X, Y, P = get_ibc_data(use_neutral=True,
                           use_subsampling=True,
                           return_subsampling_depths=True)