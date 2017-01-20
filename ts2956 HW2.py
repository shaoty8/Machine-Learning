
# coding: utf-8

# ## Homework 2 
# 
# ## ts2956 Tianyi Shao
#     ( I do not have matlab in my laptop so I am not able to use the imagesc in matlab to plot the images and I do not know how to do it with Python. So I just print the matrix for all images.:( Doing this homework with Python in such a a busy week is almost killing me and next time I'll try my best to get a Matlab!)

# ## Problem 1 (multiclass logistic regression)
# 
# 1. Data likelihood of $y_1,...,y_n$ is:
# $$ p(y_1,...,y_n|x_1,...,x_n,w_1,...,w_k)=\prod_{m=1}^{n} p(y_m|x_m,w_1,...,w_k)$$ 
# $$= \prod_{m=1}^{n}\prod_{i=1}^{k} (\frac{e^{x_m^{T}w_i}}{\sum_{j=1}^{k}e^{x_m^{T}w_j}})^{\mathbb{1}(y_m=i)}$$
# 
#  log likelihood $\mathcal{L}$ is: $log\sum_{m=1}^{n}\prod_{i=1}^{k}(\frac{e^{x_m^{T}w_i}}{\sum_{m=1}^{k}e^{x_m^{T}w_j}})^{\mathbb{1}(y=m)} = \sum_{m=1}^{n} log\prod_{i=1}^{k}(\frac{e^{x_m^{T}w_i}}{\sum_{j=1}^{k}e^{x_m^{T}w_j}})^{\mathbb{1}(y=m)} = \sum_{m=1}^{n} \prod_{i=1}^{k}(x_{m}^{T}w_i-log \sum_{j=1}^{k} e^{x_m^{T}w_j})^{\mathbb{1}(y=m)}$

# 2. $\nabla_{w_i}\mathcal{L} = \sum_{m=1}^{n} \prod_{i=1}^{k}(x_m^T - \frac{x_m^{T}e^{x_m^{T}w_i}}{\sum_{j=1}^{k} e^{x_m^{T}w_j}})^{\mathbb{1}(y=m)}$
# 
#   $\nabla_{w_i}^2\mathcal{L} = \sum_{m=1}^{n} \prod_{i=1}^{k}x_m^T(-\frac{x_m x_m^Te^{x_m^T w_j}\sum_{j=1}^{k} e^{x_m^{T}w_j} - x_m^T e^{x_m^T w_i}e^{x_m^T w_i}}{(\sum_{j=1}^{k} e^{x_m^{T}w_j})^2})^{\mathbb{1}(y=m)}$

# ## Problem 2 (Gaussian kernels)

# ## Problem 3
# ## (a)

# In[1]:

import numpy

#read all the data files
xtrainlable = numpy.loadtxt(open("label_train.txt","rb"), delimiter=",")
xtrain = numpy.loadtxt(open("Xtrain.txt","rb"), delimiter=",")
xtest = numpy.loadtxt(open("Xtest.txt","rb"), delimiter=",")
xtrue = numpy.loadtxt(open("label_test.txt","rb"), delimiter=",")
Q = numpy.loadtxt(open("Q.txt","rb"), delimiter=",")

#write a code to vote for the majority in the k_NN method
def find_majority(k):
    myMap = {}
    maximum = ( '', 0 )
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum[0]

#main lines
for k in [1,2,3,4,5]:
    
    #initialize the output vector
    testlable = numpy.zeros((500,1))
    
    #initialize the position of the corresponding xtrain digit
    ind = numpy.zeros((500,1))
    
    #for every 500 test data, compute their distances with the 5000 train data, and find k closest data
    for i in xrange(500):
        A = numpy.zeros(5000)
        for j in xrange(5000):
            for m in xrange(20):
                distance = (xtest[i,m] - xtrain[j,m])**2
            distance = numpy.sqrt(distance)
            A[j] = distance
        ind[i] = find_majority(numpy.argpartition(A, -k)[-k:])
        testlable[i] = xtrainlable[int(ind[i])]
    
    # output    
    print 'k = ', k
    print 'lable for test data:', testlable
    
    #work on the Confusion matrix
    C = numpy.zeros((10,10))
    for i in xrange(500):
        C[int(xtrue[i]),int(testlable[i])] += 1
    print 'the prediction accuracy:' , numpy.trace(C)/500.0
    
    #for k=1,3,5 find 3 mistakes
    if k % 2 != 0:
        a = 0
        B = numpy.zeros((3,20))
        predictclass = numpy.zeros(3)
        for i in xrange(500):
            if xtrue[i] != testlable[i]:
                predictclass[a] = testlable[i]
                B[a] = xtest[i]
                a += 1
                if a == 3:
                    break
        
        #Plot their images
        for i in xrange(3):
            b = numpy.zeros((20,1))
            #transpose row vector into column vector
            for j in xrange(20):
                b[j] = B[i,j]
            Q = numpy.asmatrix(Q)
            image = numpy.dot(Q,b)
            image = numpy.reshape(image, (28, 28))
            print image
            #then use the imagesc(image) in matlab to plot the image of the digit
        
    


# ## (b)

# In[6]:

#estimate for the class prior
xtrainlable = numpy.loadtxt(open("label_train.txt","rb"), delimiter=",")

for k in xrange(10):
    a = 0
    for i in xrange(5000):
        if xtrainlable[i] == k:
            a +=1
    print a 


# Part 1: multivariate Bayes classifier
# 
# 
# Suppose $X = \mathbb{R}^{20}, Y = \{0,1,...,9\}$, and the distribution $\mathcal{P}$ using a class-specific multivariate Gaussian distribution of (X,Y) is as follows:
# 
# class prior:$P(Y = y) = \pi_y,y\in \{0,1,...,9\} $
# 
# estimate for the class prior: according to the training data and code above, $P(Y = y) = \frac{1}{10},y\in \{0,1,...,9\} $
# 
# class conditional density for class $y\in \{0,1,...,9\}$ : $P_y(x) = N(x|\mu_y,\Sigma_y^2) = \frac{1}{(2\pi)^{10}\Sigma_y}exp(-\frac{1}{2}((x-\mu_y)^T\Sigma_y^{-1}(x-\mu_y)))$
# 
# Bayes classifier:
# $$f(x)^{\star} = argmax_{y\in\{0,1,...,9\}}p(X = x|Y = y)P(Y = y)$$
# $$= j,\quad  if \quad \frac{1}{\Sigma_j}exp(-\frac{1}{2}(x-\mu_y)^T\Sigma_y^{-1}(x-\mu_y))\, is\, the\, largest\, among\, all\, y\in\{0,1,..,9\}$$
# 
# maximum likelihood for data$(x_1,y_j),...,(x_n,y_j):\prod_{i=1}^{n}P_{y_j}(x) = \prod_{i=1}^{n}P(x|\mu_{y_j},\Sigma_{y_j}^2)$
# 
# the mean:  $\mu_{y_j} = \frac{1}{n}\sum_{i=1}^{n}x_i$
# 
# the covariance:  $\Sigma_{y_j} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu_{y_j})(x_i-\mu_{y_j})^T$
# 
# 

# Part 2: the confusion matrix
# 
# For a confusion matrix, each column of the matrix represents the instances in a predicted class while each row represents the instances in an actual class.
# 
# If the predicted class is different from the true class, then one of the non-diagonal elements is added by one.
# 
# If the predicted class is correct, then one of the diagonal elements is added by one.
# 
# So the sum of all diagonal elements, which is the trace of the confusion matrix, is the total number of correct predictions. Thus in part a, the prediction accuracy is the trace divided by total number of predictions made, which is 500.
# 
# To show the confusion matrix in a table, just add a string of "print C" in the code in part a.

# In[11]:

#Part 3

Q = numpy.loadtxt(open("Q.txt","rb"), delimiter=",")
Q = numpy.asmatrix(Q)

#compute the mean
for k in xrange(10):
    a = numpy.zeros((1,20))
    for i in xrange(5000):
        if xtrainlable[i] == k:
            a += xtrain[i]
    a = a/500.0
    
    #transpose a
    b = numpy.zeros((20,1))
    for j in xrange(20):
        b[j] = a[0,j]
    image = numpy.dot(Q,b)
    
    image = numpy.reshape(image, (28, 28))
    
    print image
    #then use the imagesc(image) in matlab to plot the image of the digit


# In[ ]:

#part 4
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


xtrain = numpy.loadtxt(open("Xtrain.txt","rb"), delimiter=",")
xtrainlable = numpy.loadtxt(open("label_train.txt","rb"), delimiter=",")

#construct a matrix A to store mu and sigma for each y
A = numpy.zeros((20,20))

#find mean and covariance for each y
for k in xrange(10):
    mu = numpy.zeros((1,20))
    sigma = numpy.zeros((1,20))
    for i in xrange(5000):
        if xtrainlable[i] == k:
            mu += numpy.array(xtrain[i])
        mu /= 500.0
    for i in xrange(5000):
        if xtrainlable[i] == k:
            sigma += numpy.dot(numpy.array(xtrain[i])- mu,(numpy.array(xtrain[i]) - mu).T)
        sigma /= 500.0
        A[int(2*k)] = mu
        A[int(2*k+1)] = sigma*numpy.ones(20)

#compute the class conditional probobility
for y in xrange(10):
    p = numpy.zeros((10,1))
    sigma = A[int(2*y+1),0]
    mu = A[int(2*y)]
    def p(x):
        x = numpy.array(x)
        k = (1.0/((2*numpy.pi)**10)*sigma)
        return k*numpy.exp(-0.5*numpy.dot((x-mu).T*((sigma)**(-1)),(x-mu)))
    
    Prob = numpy.zeros((500,10))
    for i in xrange(500):
        
        Prob[i,y] = p(xtest[i])

#Get the out put of test data
Testlable = numpy.zeros((500,1))
for i in xrange(500):
    for j in xrange(10):
        if Prob[i,j] == numpy.amax(Prob[i]):
            Testlable[i] = j

#plot the probibility distribution
for i in xrange(500):
    a = 0
    if xtrue[i] != Testlable[i]:
        axes = fig.add_subplot(1, 1, 1)
        axes.plot([0,1,2,3,4,5,6,7,8,9], Prob[i], 'r')
        axes.set_title("Probability Distribution over the 10 Classes")
        axes.set_xlabel("Classes")
        axes.set_ylabel("Probability")
        a += 1
        if a == 3:
            break
plt.show()

