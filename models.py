# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *



class FFNN(nn.Module):
    """
    From exmaple Fnn code
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """

    def __init__(self, word_embeddings=None, inp=50, hid=32, out=2):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """

        super(FFNN, self).__init__()
        if word_embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors),freeze=False)
        #self.embeddings = word_embeddings.get_initialized_embedding_layer()
       # self.word_embeddings.weight.requires_grad = False
        self.V = nn.Linear(inp, hid)
        self.V2 = nn.Linear(hid, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.dropout = nn.Dropout(0.25)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data; this is the sentence tensor.
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
      #  if self.word_embeddings is not None :
           
          
        mean = torch.mean(self.word_embeddings(x), dim=1, keepdim=False).float()
        
        return self.W(self.g(self.V2(self.g(self.V(mean)))))
     #   else:
      #      return self.W(self.g(self.V(x)))



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, word_embeddings: WordEmbeddings):
        SentimentClassifier.__init__(self)
        self.word_embeddings = word_embeddings
   
        self.indexer = word_embeddings.word_indexer
        self.input = self.word_embeddings.get_embedding_length()
        self.hidden = 256
        self.output = 2
        self.loss = nn.CrossEntropyLoss()
        self.model = FFNN(word_embeddings, self.input, self.hidden, self.output)

    def predict(self, ex_words: List[str],has_typos: bool):
        '''

        Args:
            ex_words: the sentence, a list of strings, or words in the sentence

        Returns:

        '''
        words_idx=[]
        for word in ex_words:
          

            words_idx.append(max(1, self.indexer.index_of(word)))
        
        words_tensor=torch.tensor([words_idx])
    
        y_probs = self.model.forward(words_tensor)
        return torch.argmax(y_probs)
    def loss(self, probs, target):
        return self.loss(probs, target)
class PrefixEmbeddings:
    """
    Use wordembeddings
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
   

   
    epochs = 15
    pad_size = 50
    batch_size = 200

    if train_model_for_typo_setting:
     
        classify = NeuralSentimentClassifier(PrefixEmbeddings(word_embeddings.word_indexer, word_embeddings.vectors))
    else:
       
        classify = NeuralSentimentClassifier(word_embeddings)
    word_indices = {}
    for i in range(len(train_exs)):
        words = train_exs[i].words
      
        index_list = []
        for word in words:
          
            idx = classify.indexer.index_of(word)
           
            index_list.append(max(idx, 1))
       
        word_indices[i] = index_list
      


    initial_learning_rate = 0.001
    optimizer = optim.Adam(classify.model.parameters(), lr=initial_learning_rate)
   
    train_indices = [idx for idx in range(0,len(train_exs))]
    
    for epoch in range(epochs):
        random.shuffle(train_indices)
 
        batch_x = []
        batch_y = []
        total_loss = 0.0
        for idx in train_indices:
            if len(batch_x) < batch_size:

                sent_pad = [0]*pad_size
                sent = word_indices[idx]
              
                sent_pad[:min(pad_size,len(sent))]=sent[:min(pad_size,len(sent))]
                batch_x.append(sent_pad)
               
                target = train_exs[idx].label
                batch_y.append(target)
            else:
                classify.model.train()
                optimizer.zero_grad()
                
                batch_x = torch.tensor(batch_x)
              
                probs =  classify.model.forward(batch_x)
                target = torch.tensor(batch_y)
                
                loss = classify.loss(probs, target)
             
                total_loss += loss
                
                loss.backward()
                
                optimizer.step()
                batch_x = []
                batch_y = []

        total_loss /= len(train_exs)

        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return classify

