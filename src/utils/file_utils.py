import pickle


def load_from_file(filename):
    with open(filename, "rb") as f:
        unpickler = pickle.Unpickler(f)
        data = unpickler.load()
    return data


def dump_to_file(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
