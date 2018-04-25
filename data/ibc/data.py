import pickle


def get_ibc_data(use_neutral=False,
                 use_subsampling=False,
                 return_subsampling_positions=False
                 label_map=[0,1,2]):


    lib, con, neutral = pickle.load(open('ibcData.pkl', 'rb'))

    if not use_neutral:

        extract = lambda x: [e.get_words() for e in x]
        lib, con, neutral = extract(lib), extract(con), extract(neutral)

        if not use_neutral:
            neutral = []

        X = [*lib, *con, *neutral]
        Y = [*[label_map[0]]*len(lib),
             *[label_map[1]]*len(con),
             *[label_map[2]]*len(neutral)]

        if return_subsampling_positions:
            raise ValueError('there is no subsampling positions if'
                             'this function is called without sub'
                             'sampling')

        return X, Y

    else:

        X, Y, P = [], [], []
        entries = [*lib, *con, *neutral]
        for tree in entries:
            for node in tree:
                X.append(node.get_words())
                Y.append(node.label)
                P.append(node.pos)

        if return_subsampling_positions:
            return X, Y, P
        return X, Y, P


