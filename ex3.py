from collections import Counter


def load_input(input_filename):
    with open(input_filename, 'r') as development_set_file:
        lines = development_set_file.readlines()

    zipped = zip(lines[0::2], lines[1::2])

    res = map(lambda x: (x[0], x[1].strip().split(" "), Counter(x[1].strip().split(" "))), zipped)

    return res


res = load_input("dataset/develop.txt")

print(res[0])
