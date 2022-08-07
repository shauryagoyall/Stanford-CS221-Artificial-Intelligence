import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

############################################################
# Custom Types
# NOTE: You do not need to modify these.

"""
You can think of the keys of the defaultdict as representing the positions in
the sparse vector, while the values represent the elements at those positions.
Any key which is absent from the dict means that that element in the sparse
vector is absent (is zero).
Note that the type of the key used should not affect the algorithm. You can
imagine the keys to be integer indices (e.g., 0, 1, 2) in the sparse vectors,
but it should work the same way with arbitrary keys (e.g., "red", "blue", 
"green").
"""
SparseVector = DefaultDict[Any, float]
Position = Tuple[int, int]


############################################################
# Problem 4a

def find_alphabetically_first_word(text: str) -> str:
    """
    Given a string |text|, return the word in |text| that comes first
    lexicographically (i.e., the word that would come first after sorting).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() handy here. If the input text is an empty string, 
    it is acceptable to either return an empty string or throw an error.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return min(word for word in text.split())
    # END_YOUR_CODE


############################################################
# Problem 4b

def euclidean_distance(loc1: Position, loc2: Position) -> float:
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    return math.sqrt((loc1[0] - loc2[0] )**2 + (loc1[1] - loc2[1] )**2)
    # END_YOUR_CODE


############################################################
# Problem 4c

def mutate_sentences(sentence: str) -> List[str]:
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be "similar" to the original sentence if
      - it has the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the
        original sentence (the words within each pair should appear in the same
        order in the output sentence as they did in the original sentence).
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more
        than once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse',
                 'the cat and the cat', 'cat and the cat and']
                (Reordered versions of this list are allowed.)
    """
    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)

    #############################
    #############################
    
    #An alternate solution is to make a dictionary mapping a word to all possible next words. Then a similar recurse function. For word in dict.keys() add one of the values and recurse. if length is met, append and go back one step and continue. check for duplicates and done
    #this would be more efficient and code length would be smaller
    
    #################################
    ################################
    
    
    similar = []
    words = sentence.split()
    num_words = len(words)
    adjacent = [0]*(num_words - 1)
    for i in range(num_words - 1):
        adjacent[i] = [ words[i], words[i+1] ]
        
    adj_words ={}
    for i in range(num_words - 1):
        word1 = adjacent[i]
        word2 = []
        for j in range(num_words - 1):
            if word1[1] == adjacent[j][0]:
                word2.append(adjacent[j])
        adj_words[word1[1]] = word2
        
    def word_maker(current_sen, similar_sentences):
        
        last_word = current_sen[-1]
        
        if len(current_sen) == num_words:
            similar_sentences.append(current_sen)
            return similar_sentences
        
        possible = adj_words[last_word]
        
        for phrase in possible:
            copy_sen = current_sen[:]
            copy_sen.append(phrase[1])
            similar_sentences = word_maker(copy_sen, similar_sentences)
        
        return similar_sentences
    
    for word in adjacent:
        similar_sentences = word_maker(word, [])
        if len(similar_sentences) != 0:
            for sentence_list in similar_sentences:
                sentence = ""
                for i in sentence_list:
                    sentence += i + " "
                sentence = sentence[:-1]
                if not sentence in similar:
                    similar.append(sentence)
    return similar   
    
    # END_YOUR_CODE


############################################################
# Problem 4d

def sparse_vector_dot_product(v1: SparseVector, v2: SparseVector) -> float:
    """
    Given two sparse vectors (vectors where most of the elements are zeros)
    |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.

    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    Note: A sparse vector has most of its entries as 0.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return sum( v1[i] * v2[i] for i in v1)
    # END_YOUR_CODE


############################################################
# Problem 4e

def increment_sparse_vector(v1: SparseVector, scale: float, v2: SparseVector,
) -> None:
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    If the scale is zero, you are allowed to modify v1 to include any
    additional keys in v2, or just not add the new keys at all.

    NOTE: This function should MODIFY v1 in-place, but not return it.
    Do not modify v2 in your implementation.
    This function will be useful later for linear classifiers.
    """
    
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in v2.keys():
        v1[key] += scale * v2[key]
        #if there is no key-value term in v1, it is created as 0.0 because defacultdict(float) and then the v2[key] value is scaled and added to it
    # END_YOUR_CODE


############################################################
# Problem 4f

def find_nonsingleton_words(text: str) -> Set[str]:
    """
    Split the string |text| by whitespace and return the set of words that
    occur more than once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    count = collections.defaultdict(int)
    words = text.split()
    for k in words :
        count[k] += 1
    return set(word for word in count if count[word] > 1)
    # END_YOUR_CODE
