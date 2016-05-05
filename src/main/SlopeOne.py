#coding: utf-8
from collections import defaultdict
import math
class SlopeOne(object):
    def __init__(self, train_file, test_file):
        self.diffs = defaultdict(dict)
        self.freqs = defaultdict(dict)
        self.train_file = train_file
        self.test_file = test_file
        self.train = defaultdict(dict)
        self.test = defaultdict(dict)
        self.loadData()
        self.compute_diff(self.train)

    def loadData(self):
        with open(self.train_file) as f:
            for line in f:
                user, item, score, _ = line.strip().split("\t")
                self.train[user][item] = int(score)
        with open(self.test_file) as f:
            for line in f:
                user, item, score, _ = line.strip().split("\t")
                self.test[user][item] = int(score)

    def predict(self, user, item):
        pred = 0.0
        freq = 0
        for it, score in self.train[user].iteritems():
            try:
                f = self.freqs[it][item]
            except KeyError:
                continue
            pred += self.freqs[it][item] * (score - self.diffs[it][item])
            freq += self.freqs[it][item]
        result = pred / freq if freq > 0 else 3
        if result > 5:
            result = 5
        return result

    def compute_rmse(self):
        rmse = 0
        count = 0

        for user, ratings in self.test.iteritems():
            for item, score in ratings.iteritems():
                pred = self.predict(user, item)
                #print item, score, pred
                rmse += (score - pred) ** 2
                count += 1
        return math.sqrt(rmse * 1.0 / count)

    def compute_diff(self, data):
        for ratings in data.itervalues():
            for a, sa in ratings.iteritems():
                for b, sb in ratings.iteritems():
                    self.freqs[a].setdefault(b, 0)
                    self.diffs[a].setdefault(b, 0.0)
                    self.freqs[a][b] += 1
                    self.diffs[a][b] += sa - sb

        for a, ratings in self.diffs.iteritems():
            for b in ratings:
                ratings[b] /= self.freqs[a][b]

if __name__ == '__main__':
    #s = SlopeOne('../../ml-100k/train.base', '../../ml-100k/test.base')
    s = SlopeOne('../../ml-1m/train.base', '../../ml-1m/test.base')
    print s.compute_rmse()
