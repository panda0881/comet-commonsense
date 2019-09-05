import pickle

with open('results/O_beam_10.pickle', 'rb') as f:
    test_data = pickle.load(f)

relation_count = dict()
for tmp_example in test_data:
    if tmp_example['r'] not in relation_count:
        relation_count[tmp_example['r']] = 0
    relation_count[tmp_example['r']] += 1

print('end')
