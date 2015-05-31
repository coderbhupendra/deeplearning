__docformat__ = 'restructedtext en'

from dataset import Dictionary, load_corpus

import cPickle
import numpy as np
import theano
import theano.tensor as T
import math
from future_builtins import zip
import time
import logging



class LogBilinearLanguageModel(object):
    """
    Log-bilinear language model class
    context are the previous words which we give as input inorder to predict the next word from vocablary
    V is the size of vocablary
    K if the size of feature vector for word here its is 20
    content_sz is the no of previos words we need to predict the next words , here its 2
    """


    def __init__(self, context, V, K, context_sz, rng):
        """
        Initialize the parameters of the language model
        R is the look up vocablary matrix which gives feature vector for each word
        Q is weight matrix for upper layer between hidden layer and output layer of size V*K 
        C is the weight matrix between input layer and hidden layer ,here it is 3D type , i.e. (context_sz, K, K)
        b is the bias for output layer
        r_w is the concatenated input for the set of word , i.e if we give 2 words , it will give concatenated feature vector for both words
        q_hat is the dot product for weight matrix C and input r_w
        then we apply tanh() to it , which becomes the hidden layer
        s is the dot product of weight matrix Q and q_hat
        p_w_given_h is the softmax on s , i.e p_w_given_h gives the probability for each word in vocablary and its teh final output
        """

        """
        now how did i convert the words into ids, 
        i actually did know how to do it , so i got it from a git project ,
        where they where reading the any text file and convert it into vocablary and corpus
        and assigning ids to every word in asecending numbers
        and for every word i am having feature vector of size 20
        and once i have feature vector its easy to do feed forward in neural network 
        and for backpropogation i used the general method of theano , i.e. T.gparam(cost,params)
        and updated weights
        """
        # training contexts
        self.context = context
        print"type x ",(type(context))
        print"x",(context)
        # initialize context word embedding matrix R of shape (V, K)
        # TODO: parameterize initialization
        R_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)

        self.R = theano.shared(value=R_values, name='R', borrow=True)
        # initialize target word embedding matrix Q of shape (V, K)
        Q_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)
        self.Q = theano.shared(value=Q_values, name='Q', borrow=True)
        # initialize weight tensor C of shape (context_sz, K, K)
        C_values = np.asarray(rng.normal(0, math.sqrt(0.1), 
                                         size=(context_sz, K, K)), 
                              dtype=theano.config.floatX)
        self.C = theano.shared(value=C_values, name='C', borrow=True)
        # initialize bias vector 
        b_values = np.asarray(rng.normal(0, math.sqrt(0.1), size=(V,)), 
                              dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        # context word representations
        self.r_w = self.R[context]
        # predicted word representation for target word
        self.q_hat = T.tensordot(self.C, self.r_w, axes=[[0,1], [1,2]])
        self.q_hat=T.tanh(self.q_hat)
        # similarity score between predicted word and all target words
        self.s = T.transpose(T.dot(self.Q, self.q_hat) + T.reshape(self.b, (V,1)))
        # softmax activation function
        self.p_w_given_h = T.nnet.softmax(self.s)
        # parameters of the model
        self.params = [self.R, self.Q, self.C, self.b]
        
        
    def negative_log_likelihood(self, y):
        # take the logarithm with base 2
        print"type y ",(type(y))
        print"y",(y)
        return -T.mean(T.log2(self.p_w_given_h)[T.arange(y.shape[0]), y])

        
def make_instances(corpus, vocab, context_sz, start_symb='<s>', end_symb='</s>'):
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=np.int32), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=np.int32), borrow=borrow)
        return shared_x, shared_y
    data = []
    labels = []        
    for sentence in corpus:
        # add 'start of sentence' and 'end of sentence' context
        sentence = [start_symb] * context_sz + sentence + [end_symb] * context_sz
        sentence = vocab.doc_words_to_ids(sentence, update_dict=False)
        for instance in zip(*(sentence[i:] for i in xrange(context_sz+1))):
            data.append(instance[:-1])
            labels.append(instance[-1])
    print "data",(data[1:10])
    print (type(data))
    print(np.asarray(data).shape)
    print "labels",(labels[1:10])
    print(type(labels))
    print(np.asarray(labels).shape)        
    train_set_x, train_set_y = shared_dataset([data, labels])
    return train_set_x, train_set_y


    

    
def train_lbl(train_data='train', dev_data='dev', test_data='test', 
              K=20, context_sz=2, learning_rate=1.0, 
              rate_update='simple', epochs=10, 
              batch_size=100, rng=None, patience=None, 
              patience_incr=2, improvement_thrs=0.995, 
              validation_freq=1000):
    """
    Train log-bilinear model
    """
    # create vocabulary from train data, plus <s>, </s>
    vocab = Dictionary.from_corpus(train_data, unk='<unk>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    V = vocab.size()

# create random number generator

    #rng = np.random.RandomState(123)

    # initialize random generator if not provided
    rng = np.random.RandomState() if not rng else rng
    

    # load data
    print("Load data ...")
    
    with open('train' , 'rb') as fin:
        train_data = [line.split() for line in fin.readlines() if line.strip()]  
        print "train_data ",(np.array(train_data).shape)
        print(type(train_data))
    with open('dev', 'rb') as fin:
        dev_data = [line.split() for line in fin.readlines() if line.strip()]
    
    with open('test', 'rb') as fin:
            test_data = [line.split() for line in fin.readlines() if line.strip()]
   

    # generate (context, target) pairs of word ids
    train_set_x, train_set_y = make_instances(train_data, vocab, context_sz)
    dev_set_x, dev_set_y = make_instances(dev_data, vocab, context_sz)
    test_set_x, test_set_y = make_instances(test_data, vocab, context_sz)
    print "train_set_x",(train_set_x.eval())
    print (type(train_set_x.eval()))
    print(train_set_x[:100].eval().shape)
    print "train_set_y",(train_set_y.eval())
    print (type(train_set_y.eval()))
    print(train_set_y[:100].eval().shape)


    # number of minibatches for training
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_dev_batches = dev_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # build the model
    print("Build the model ...")
 
    index = T.lscalar()
    x = T.imatrix('x')
    y = T.ivector('y')
    # create log-bilinear model
    lbl = LogBilinearLanguageModel(x, V, K, context_sz, rng)

    # cost function is negative log likelihood of the training data
    cost = lbl.negative_log_likelihood(y)
    # compute the gradient
    gparams = []
    for param in lbl.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameter of the model
    updates = []
    for param, gparam in zip(lbl.params, gparams):
        updates.append((param, param-learning_rate*gparam))

    # function that computes log-probability of the dev set
    logprob_dev = theano.function(inputs=[index], outputs=cost,
                                  givens={x: dev_set_x[index*batch_size:
                                                           (index+1)*batch_size],
                                          y: dev_set_y[index*batch_size:
                                                           (index+1)*batch_size]
                                          })


    # function that computes log-probability of the test set
    logprob_test = theano.function(inputs=[index], outputs=cost,
                                   givens={x: test_set_x[index*batch_size:
                                                             (index+1)*batch_size],
                                           y: test_set_y[index*batch_size:
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
    print("training model...")
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

    return lbl

    
if __name__ == '__main__':
    train_lbl()
    

   
   
       
        