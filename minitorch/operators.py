"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y

def id(x: float) -> float:
    "$f(x) = x$"
    return x

def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    return x + y

def neg(x: float) -> float:
    "$f(x) = -x$"
    return -x

def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    return 1.0 if abs(x - y) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    Calculate as:
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >= 0 else $\frac{e^x}{(1.0 + e^{x})}$ for stability.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    "$f(x) =$ x if x is greater than 0, else 0"
    return max(0, x)

EPS = 1e-6

def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)

def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    return d / (x + EPS)

def inv(x: float) -> float:
    "$f(x) = 1/x$"
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    return -d / (x ** 2)

def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    return d if x > 0 else 0.0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


from typing import Callable, Iterable
from functools import reduce as py_reduce

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    Applies `fn` to each element of the input iterable and returns a new iterable.
    """
    def mapper(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return mapper

def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negates each element in the list using `map`.
    """
    return map(lambda x: -x)(ls)

def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipWith.

    Combines elements from two iterables using `fn`.
    """
    def zipper(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return zipper

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Adds elements of `ls1` and `ls2` using `zipWith` and `add`.
    """
    return zipWith(lambda x, y: x + y)(ls1, ls2)

def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """
    Higher-order reduce.

    Reduces an iterable to a single value using `fn` and an initial `start` value.
    """
    def reducer(ls: Iterable[float]) -> float:
        return py_reduce(fn, ls, start)

    return reducer

def sum(ls: Iterable[float]) -> float:
    """
    Sums up all elements in the list using `reduce`.
    """
    return reduce(lambda x, y: x + y, 0)(ls)

def prod(ls: Iterable[float]) -> float:
    """
    Calculates the product of all elements in the list using `reduce`.
    """
    return reduce(lambda x, y: x * y, 1)(ls)