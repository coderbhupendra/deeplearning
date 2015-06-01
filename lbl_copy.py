__docformat__ = 'restructedtext en'

from dataset import Dictionary, load_corpus
import nltk, re, pprint,urllib2
from nltk import word_tokenize


import cPickle
import numpy as np
import theano
import theano.tensor as T
import math
from future_builtins import zip
import time
import logging



class NeuralLanguageModel(object):
    """
    Log-bilinear language model class
    context are the previous words which we give as input inorder to predict the next word from vocablary
    Vocal_size is the size of vocablary
    K_feature if the size of feature vector for word here its is 20
    content_sz is the no of previos words we need to predict the next words , here its 2

    """


    def __init__(self, x, Vocal_size, K_feature, hidden_units,context_sz, rng,batch_size):
        """
        Initialize the parameters of the language model
 x is a of the form                            :::::               y is of the form 
                                                                   [   3    6    9 ..., 8208 8211 8214]
[[  21  995  221]
 [1498  161   17]
 [  22 1211  922]
 ..., 
 [ 100 1393 1360]
 [1140  227  684]
 [1321 1408  922]]

        'look_up' is the look up vocablary matrix which gives feature vector for each word

        'upper_weights' is weight matrix for upper layer between hidden layer and output layer 
        'b' is the bias for output layer
        
        'lower_weights' is the weight matrix between input layer and hidden layer 
        'd' is the bias for hidden layer
        
        'r_w' is the concatenated input for the set of word , i.e if we give 2 words , 
        it will give concatenated feature vector for both words
        
        'q_hat' is the dot product for weight matrix 'lower_weights' and input 'r_w'
        then we apply tanh() to it , which becomes the hidden layer

        's' is the dot product of weight matrix 'upper_weights' and 'q_hat'
        'p_w_given_h' is the softmax on 's' , i.e 'p_w_given_h' gives the probability for each word in vocablary and its teh final output

        """

       
        # 'x'training contexts
        self.x = x

        # TODO: parameterize initialization
        look_up_matrix=np.asarray(rng.uniform(-2, 2, size=(Vocal_size, K_feature)),dtype=theano.config.floatX)
        self.look_up = theano.shared(value=look_up_matrix, name='look_up', borrow=True)

        
         # initialize  matrix 'lower weights' of shape (hidden_units,context_sz*K_feature) and initialize bias vector 'd'
        l_weights = np.asarray(rng.normal(0, math.sqrt(2),size=(hidden_units,context_sz*K_feature)), dtype=theano.config.floatX)
        self.lower_weights = theano.shared(value=l_weights, name='lower_weights', borrow=True)
        
        d_values = np.asarray(rng.normal(0, math.sqrt(2), size=(hidden_units,)),dtype=theano.config.floatX)
        self.d = theano.shared(value=d_values, name='d', borrow=True)

       
        
        # initialize  matrix 'upper weights' of shape (Vocal_size, hidden_units) and initialize bias vector 'b'
        u_weights = np.asarray(rng.uniform(-2, 2, size=(Vocal_size, hidden_units)),dtype=theano.config.floatX)
        self.upper_weights = theano.shared(value=u_weights, name='upper_weights', borrow=True)
         
        b_values = np.asarray(rng.normal(0, math.sqrt(2), size=(Vocal_size,)),dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        
        
        
        # context word representations
        self.r_w =T.reshape( self.look_up[x] , (batch_size,(K_feature*context_sz)) )
      
       
        # predicted word representation for target word
        self.q_hat = T.dot(self.r_w, self.lower_weights.T) +T.reshape(self.d, (1,hidden_units))
        self.q_hat=T.tanh(self.q_hat)
        
        # similarity score between predicted word and all target words
        self.s =T.dot(self.q_hat,(self.upper_weights.T)) +T.reshape(self.b, (1,Vocal_size))
        # softmax activation function
        self.p_w_given_h = T.nnet.softmax(self.s) 
        
        # parameters of the model
        self.params = [self.look_up, self.upper_weights, self.lower_weights,self.d,self.b]
        
        
    def negative_log_likelihood(self, y):
        # take the logarithm with base 2
        return -T.mean(T.log2(self.p_w_given_h)[T.arange(y.shape[0]), y])



def make_instances(text_tokenized, dictionary, context_sz):
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy

       # print "datax\n",(data_x)
        shared_x = theano.shared(value=np.asarray(data_x, dtype=np.int32), borrow=borrow)
        shared_y = theano.shared(value=np.asarray(data_y, dtype=np.int32), borrow=borrow)
        #print "shared x",(shared_x.get_value(borrow=True))
        return shared_x, shared_y

    print(len(text_tokenized))
    print(len(dictionary))
    data = []
    labels = []   
    token=0     
    no_tokens=len(text_tokenized)
    oov_no=len(dictionary)
    #print(text_tokenized)
    for token in range(0,no_tokens,context_sz):
        

        sentence=text_tokenized[token:token+context_sz]
        #print(sentence)
        #print(token)
        token+=context_sz
        if (token<len(text_tokenized)):
            concatenated_context=list()
            for i in range(context_sz):
                
                                        if(sentence[i] in dictionary.keys()): 
                                                concatenated_context.append( dictionary[sentence[i]])
                                        else:
                                                concatenated_context.append( dictionary["oov"])
        
            data.append(np.array(concatenated_context).flatten()) 
            """ this data is in the form of concatenated nos for respective words of sentence"""
            if(text_tokenized[token] in dictionary.keys()): 
                                                #if i want my yout put word to be in fecture vector form uncomment it , 
                                                #else y contains number for the prediction context 
                                                labels.append(dictionary[text_tokenized[token]])
                                                #labels.append(token)
            else:
                                                labels.append(dictionary["oov"])
                                                #labels.append(no_tokens)

        #if (token<len(text_tokenized-1)):
        token+=1
    
    data=np.asarray(data)  
    
    
   
    #print(data)
    #print(labels)
    labels= ( (np.array(labels)).flatten())
    #print(data)
    #print(labels)
    train_set_x, train_set_y = shared_dataset([data, labels])
  
    #print('ss')
    #print(train_set_x.get_value(borrow=True))
    return train_set_x, train_set_y


    

    
def train_nbl(train_data='train', dev_data='dev', test_data='test', 
              K=5,hidden_units=20, context_sz=3, learning_rate=1.0, 
              rate_update='simple', epochs=10, 
              batch_size=100, rng=None, patience=None, 
              patience_incr=2, improvement_thrs=0.995, 
              validation_freq=20):
    """
    Train neural model
    """
   
    
    
# create random number generator
    #rng = np.random.RandomState(123)

    # initialize random generator if not provided
    rng = np.random.RandomState() if not rng else rng
    

    # load data
    print("Load data ...")
    #url = "http://www.gutenberg.org/files/2554/2554.txt"
    #edgeworth-parents.txtresponse = urllib2.urlopen(url)
    #raw = response.read()
    

    """ here i am reading text file and tokenzing it """
    #large :edgeworth-parents.txt small :blake-poems.txt
    with open('blake-poems.txt', 'rb') as fin:
        raw = fin.read()
    tokens=word_tokenize(raw)
    text = nltk.Text(tokens)
    words = [w.lower() for w in tokens]
    train_data=words
    
    vocab_data=sorted(set(words))
    vocab_data.append("oov")
    size=len(vocab_data)
    no=range(size)
    print('size of vocab dictionary %i'%size)
    print('size of training data %i'%len(text))
    


    vocab_dic = zip(vocab_data,no)
    dictionary=dict(vocab_dic)
    #print(dictionary['how'])
    #large :'austen-emma.txt' small:burgess-busterbrown.txt
    with open('burgess-busterbrown.txt', 'rb') as fin:
        raw_d = fin.read()
    tokens=word_tokenize(raw_d)
    text = nltk.Text(tokens)
    words = [w.lower() for w in tokens]
    dev_data = words
    #lareg :'melville-moby_dick.txt' small:shakespeare-macbeth.txt
    with open('shakespeare-macbeth.txt', 'rb') as fin:
        raw_t = fin.read()
    tokens=word_tokenize(raw_t)
    text = nltk.Text(tokens)
    words = [w.lower() for w in tokens]
    test_data = words
   

    """ here i am giving set of words which i made in above steps to fuction make_instance 
        and it will return me context in the form of set of numbers representing the context like [152,456,85] and the next word like[45] """
    # generate (context, target) pairs of word ids
    train_set_x, train_set_y = make_instances(train_data, dictionary, context_sz)
    dev_set_x, dev_set_y = make_instances(dev_data, dictionary, context_sz)
    test_set_x, test_set_y = make_instances(test_data, dictionary, context_sz)
   
    print "\n\ntrain_set_x\n",(train_set_x.eval())
    #print (type(train_set_x.eval()))
    print "shape of x",(train_set_x.eval().shape)
    print "\ntrain_set_y\n",(train_set_y.eval())
    #print (type(train_set_y.eval()))
    print "shape of y ",(train_set_y.eval().shape)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_dev_batches = dev_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print('no of batches in training data%i'%n_train_batches)
    #print('shape of dataset xrange')
    #print(train_set_x.get_value(borrow=True).shape)
    #print(train_set_x.get_value(borrow=True)[0:50])
    #print('shape of dataset y')
    #print(train_set_y.get_value(borrow=True).shape)
    # build the model
    print("\nBuild the model ...")
 
    index = T.lscalar()
    x = T.imatrix('x')
    y = T.ivector('y')
    # create log-bilinear model
    # size is the size of vocab dict 
    
    nbl = NeuralLanguageModel(x, size, K,hidden_units, context_sz, rng,batch_size)
    
    # cost function is negative log likelihood of the training data
    cost = nbl.negative_log_likelihood(y)
    # compute the gradient
    gparams = []
    for param in nbl.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameter of the model
    updates = []
    for param, gparam in zip(nbl.params, gparams):
        updates.append((param, param-learning_rate*gparam))

    # function that computes log-probability of the dev set
    

    # function that computes log-probability of the test set
    logprob_test = theano.function(inputs=[index], outputs=cost,
                                   givens={x: train_set_x[index*batch_size:
                                                             (index+1)*batch_size],
                                           y: train_set_y[index*batch_size:
                                                             (index+1)*batch_size]
                                           })
    
    logprob_dev = theano.function(inputs=[index], outputs=cost,
                                  givens={x: dev_set_x[index*batch_size:
                                                           (index+1)*batch_size],
                                          y: dev_set_y[index*batch_size:
                                                           (index+1)*batch_size]
                                          })

    # function that returns the cost and updates the parameter 
    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={x: train_set_x[index*batch_size:
                                                             (index+1)*batch_size],
                                          y: train_set_y[index*batch_size:
                                                             (index+1)*batch_size]
                                          })

    # perplexity functions
    def compute_dev_logp():
        return np.mean([logprob_dev(i) for i in xrange(n_dev_batches)])

    def compute_test_logp():
        return np.mean([logprob_test(i) for i in xrange(n_test_batches)])

    def ppl(neg_logp):
        return np.power(2.0, neg_logp)

    # train model
    print("\ntraining model...")
    best_params = None
    last_epoch_dev_ppl = np.inf
    best_dev_ppl = np.inf
    test_ppl = np.inf
    test_core = 0
    start_time = time.clock()
    done_looping = False
    
    
    for epoch in xrange(epochs):
        

        if done_looping:
            break
        print('epoch %i' % epoch) 
        for minibatch_index in xrange(n_train_batches):

            itr = epoch * n_train_batches + minibatch_index
            train_logp = train_model(minibatch_index)
            print('epoch %i, minibatch %i/%i, train minibatch log prob %.4f ppl %.4f' % 
                         (epoch, minibatch_index+1, n_train_batches, 
                          train_logp, ppl(train_logp)))
            if (itr+1) % validation_freq == 0:
                # compute perplexity on dev set, lower is better
                dev_logp = compute_dev_logp()
                dev_ppl = ppl(dev_logp)
                print('epoch %i, minibatch %i/%i, dev log prob %.4f ppl %.4f' % 
                             (epoch, minibatch_index+1, n_train_batches, 
                              dev_logp, ppl(dev_logp)))
                # if we got the lowest perplexity until now
                if dev_ppl < best_dev_ppl:
                    # improve patience if loss improvement is good enough
                    if patience and dev_ppl < best_dev_ppl * improvement_thrs:
                        patience = max(patience, itr * patience_incr)
                    best_dev_ppl = dev_ppl
                    test_logp = compute_test_logp()
                    test_ppl = ppl(test_logp)
                    print('epoch %i, minibatch %i/%i, test log prob %.4f ppl %.4f' % 
                                 (epoch, minibatch_index+1, n_train_batches, 
                                  test_logp, ppl(test_logp)))
            # stop learning if no improvement was seen for a long time
            if patience and patience <= itr:
                done_looping = True
                break
        # adapt learning rate
        if rate_update == 'simple':
            # set learning rate to 1 / (epoch+1)
            learning_rate = 1.0 / (epoch+1)
        elif rate_update == 'adaptive':
            # half learning rate if perplexity increased at end of epoch (Mnih and Teh 2012)
            this_epoch_dev_ppl = ppl(compute_dev_logp())
            if this_epoch_dev_ppl > last_epoch_dev_ppl:
                learning_rate /= 2.0
            last_epoch_dev_ppl = this_epoch_dev_ppl
        elif rate_update == 'constant':
            # keep learning rate constant
            pass
        else:
            raise ValueError("Unknown learning rate update strategy: %s" %rate_update)
        
    end_time = time.clock()
    total_time = end_time - start_time
    print('Optimization complete with best dev ppl of %.4f and test ppl %.4f' % 
                (best_dev_ppl, test_ppl))
    print('Training took %d epochs, with %.1f epochs/sec' % (epoch+1, 
                float(epoch+1) / total_time))
    print("Total training time %d days %d hours %d min %d sec." % 
                (total_time/60/60/24, total_time/60/60%24, total_time/60%60, total_time%60))
    # return 

    return nbl

    
if __name__ == '__main__':
    train_nbl()
    

   
   
       
        
        

