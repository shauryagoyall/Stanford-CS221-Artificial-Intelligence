import collections
import math
from typing import Set, Callable, List, Iterable, Iterator, Tuple

SENTENCE_BEGIN = '-BEGIN-'


def sliding(xs: List[str], windowSize: int) -> Iterator[str]:
    for i in range(1, len(xs) + 1):
        yield xs[max(0, i - windowSize):i]


def removeAll(s: str, chars: Iterable[str]) -> str:
    return ''.join([c for c in s if c not in chars])


def alphaOnly(s: str) -> str:
    s = s.replace('-', ' ')
    return ''.join([c for c in s if c.isalpha() or c == ' '])


def cleanLine(l: str) -> str:
    return alphaOnly(l.strip().lower())


def words(l: str) -> List[str]:
    return l.split()


############################################################
# Make an n-gram model of words in text from a corpus.

def makeLanguageModels(path: str) -> Tuple[Callable[[str], float], Callable[[str, str], float]]:
    unigramCounts = collections.Counter()
    totalCounts = 0
    bigramCounts = collections.Counter()
    bitotalCounts = collections.Counter()
    VOCAB_SIZE = 600000
    LONG_WORD_THRESHOLD = 5
    LENGTH_DISCOUNT = 0.15

    def bigramWindow(win: str) -> Tuple[str, str]:
        assert len(win) in [1, 2]
        if len(win) == 1:
            return SENTENCE_BEGIN, win[0]
        else:
            return tuple(win)

    with open(path, 'r') as f:
        for l in f:
            ws = words(cleanLine(l))
            unigrams = [x[0] for x in sliding(ws, 1)]
            bigrams = [bigramWindow(x) for x in sliding(ws, 2)]
            totalCounts += len(unigrams)
            unigramCounts.update(unigrams)
            bigramCounts.update(bigrams)
            bitotalCounts.update([x[0] for x in bigrams])

    def unigramCost(x: str) -> float:
        if x not in unigramCounts:
            length = max(LONG_WORD_THRESHOLD, len(x))
            return -(length * math.log(LENGTH_DISCOUNT) + math.log(1.0) - math.log(VOCAB_SIZE))
        else:
            return math.log(totalCounts) - math.log(unigramCounts[x])

    def bigramModel(a: str, b: str) -> float:
        return math.log(bitotalCounts[a] + VOCAB_SIZE) - math.log(bigramCounts[(a, b)] + 1)

    return unigramCost, bigramModel


def logSumExp(x: float, y: float) -> float:
    lo = min(x, y)
    hi = max(x, y)
    return math.log(1.0 + math.exp(lo - hi)) + hi


def smoothUnigramAndBigram(unigramCost: Callable[[str], float], bigramModel: Callable[[str, str], float], a: float):
    """Coefficient `a` is Bernoulli weight favoring unigram"""

    # Want: -log( a * exp(-u) + (1-a) * exp(-b) )
    #     = -log( exp(log(a) - u) + exp(log(1-a) - b) )
    #     = -logSumExp( log(a) - u, log(1-a) - b )

    def smoothModel(w1: str, w2: str) -> float:
        u = unigramCost(w2)
        b = bigramModel(w1, w2)
        return -logSumExp(math.log(a) - u, math.log(1 - a) - b)

    return smoothModel


############################################################
# Make a map for inverse lookup of words without vowels -> possible
# full words

def makeInverseRemovalDictionary(path: str, removeChars: Iterable[str]) -> Callable[[str], Set[str]]:
    wordsRemovedToFull = collections.defaultdict(set)

    with open(path, 'r') as f:
        for l in f:
            for w in words(cleanLine(l)):
                wordsRemovedToFull[removeAll(w, removeChars)].add(w)

    wordsRemovedToFull = dict(wordsRemovedToFull)

    def possibleFills(short: str) -> Set[str]:
        return wordsRemovedToFull.get(short, set())

    return possibleFills
