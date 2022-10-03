#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    tokens = x.split()
    feature_vector = {}
    for word in tokens:
        try:
            feature_vector[word] +=1
        except:
            feature_vector[word] = 1
    return feature_vector
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight
    
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)

    def predictor(x):
        ''' Takes in a review in feature vector form and outputs if it is positive or negative based on the weights learned so far '''
        if dotProduct(featureExtractor(x), weights) >= 0:
            return 1
        else: 
            return -1
    
    for i in range(numEpochs):
        for review in trainExamples:
            x, y = review[0], review[1]
            #print(review)
            # x is the review
            # y is whether it is positive or negative
        
            features_dict = featureExtractor(x) #get the feature vector of the review
            hinge = dotProduct(features_dict, weights)*y #assumes weight is 0 if it has not seen the word before

            if hinge < 1.0: 
            #if hinge <1 ie loss is more than 0, do a stochastic gradient update 
                for feature in features_dict:
                    if feature not in weights:
                        weights[feature] = 0
                        
                ###########################        
                #### Alternative 1 uses increment function from util.py but is slightly slower than 
                #### than alterantive 2. Uncomment the code to switch between them
                
                ### #Alternative 1
                
                # grad = {}
                # increment(grad, -y, features_dict)
                # increment(weights, - eta, grad)
                
                ### #Alternative 2
                    gradient = - features_dict[feature]*y
                    weights[feature] = weights[feature] - eta*(gradient)
                    
    #print(f"\nEpoch{i}")
    print("Train loss: ", evaluatePredictor(trainExamples, predictor) )
    print("Validation loss: ", evaluatePredictor(validationExamples, predictor) )
        
    print("\n")
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = { random.choice(list(weights.keys())) : random.randint(-100, 100)   for i in range(random.randint(2, 30))     }
        if dotProduct(phi, weights) >= 0:
            y = 1 
        else:
            y = -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        x = x.replace(" ", "")
        features = {}
        for i in range(len(x) - n + 1):
            try:
                features[x[i:i+n]] += 1
            except:
                features[x[i:i+n]] = 1
            
        
        return features
            
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))

############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    def get_distance(d1, d2):
        """
        @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
        @param dict d2: same as d1
        @return float: the L2 norm between d1 and d2
        """
        return sum((d1.get(f, 0) - v)**2 for f, v in list(d2.items()))
      
    centers = random.sample(examples, K)
    num_examples = len(examples)
    
    for epoch in range(maxEpochs):
        allocated_center = [None for x in range(num_examples)]
        
        ### allocate centers
        
        for i in range(num_examples):
            best_distance = float("inf")
            
            for j in range(K):
                distance = get_distance(centers[j], examples[i])
                
                if distance < best_distance:
                    best_distance = distance
                    allocated_center[i] = j

        ### calculate new centers
        
        new_centers = [DefaultDict(int) for x in range(K)]
        count = [DefaultDict(int) for x in range(K)] #each feature vector in a cluster as its own count
        
        for i in range(num_examples):
            center = new_centers[allocated_center[i]]
            
            for a ,b in list(examples[i].items()):
                new_count = count[allocated_center[i]][a] + 1 #new count for a particular feature vector in a cluster
                center[a] = (center[a]*count[allocated_center[i]][a] + b)/new_count
                count[allocated_center[i]][a] = new_count
        
        if centers == new_centers:
            break
        else:
            centers = new_centers
            
    def reconstruction_loss():
        loss = 0
        for i in range(num_examples):
            loss += get_distance(centers[allocated_center[i]], examples[i])
        return loss
    
    return (
        centers,
        allocated_center,
        reconstruction_loss()
        )

    # END_YOUR_CODE
    
    
    # i do center as d1 and examples as d2 because of the filler words \
    # which only serve as noise to deviate from a center. Center will have all possible \
    # feature vectors which includes filler words. Hence by only considering the filler words  \
    # for each example (which should give loss 0) while calculating the loss and not the center's filler words, \
    # an accurate reconstruction loss is obtained
    
if __name__ == "__main__":

    for i in range(10):
        print("\nN is", i)
        testValuesOfN(i)
        print("\n")