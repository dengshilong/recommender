#coding: utf-8
from pyspark import SparkContext
from collections import defaultdict
import math
def compute_diff(values):
    diff = defaultdict(dict)
    freq = defaultdict(dict)
    for a, sa in values:
        for b, sb in values:
            freq[a].setdefault(b, 0)
            diff[a].setdefault(b, 0.0)
            freq[a][b] += 1
            diff[a][b] += sa - sb
    return (diff,freq)

def predict(user, item, diffs, freqs, train):
    pred = 0.0
    freq = 0
    for it, score in train[user]:
        try:
            f = freqs[it][item]
        except KeyError:
            continue
        pred += freqs[it][item] * (score - diffs[it][item])
        freq += freqs[it][item]
    result = pred / freq if freq > 0 else 3
    if result > 5:
        result = 5
    return result
def compute_rmse(diffs, freqs, train, test):
    rmse = 0
    count = 0

    for user, item, score in test:
        pred = predict(user, item, diffs, freqs, train)
        #print item, score, pred
        rmse += (score - pred) ** 2
        count += 1
    return math.sqrt(rmse * 1.0 / count)

if __name__ == "__main__":
    sc = SparkContext("local", "Simple Recommend")
    train_data = sc.textFile("../../ml-100k/train.base")
    test_data = sc.textFile("../../ml-100k/test.base")
    # train_data = sc.textFile("../../ml-1m/train.base")
    # test_data = sc.textFile("../../ml-1m/test.base")
    train_data = train_data.map(lambda line: line.split('\t')).\
        map(lambda tokens: (int(tokens[0]), (int(tokens[1]), int(tokens[2])))).groupByKey()
    test_data = test_data.map(lambda line: line.split('\t')).\
        map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).collect()

    data = train_data.map(lambda tokens: compute_diff(tokens[1])).collect()
    diffs = defaultdict(dict)
    freqs = defaultdict(dict)
    for diff, freq in data:
        for a in diff:
            for b in diff:
                diffs[a].setdefault(b, 0.0)
                freqs[a].setdefault(b, 0)
                diffs[a][b] += diff[a][b]
                freqs[a][b] += freq[a][b]
    for x in diffs.keys():
        for y in diffs.keys():
            try:
                f = freqs[x][y]
            except KeyError:
                continue
            if freqs[x][y] != 0:
                diffs[x][y] = diffs[x][y] * 1.0 / freqs[x][y]
    print 'ok'
    data = None
    train_data = train_data.collectAsMap()
    print compute_rmse(diffs, freqs, train_data, test_data)
