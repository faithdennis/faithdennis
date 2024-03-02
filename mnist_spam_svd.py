#!/usr/bin/env python
# coding: utf-8

# # HW 1: Intro

# #### First we import our modules and data and seed our RandomGenerator for reproducibility

# In[1]:


from sklearn import svm
import matplotlib.pyplot as plt

from save_csv import *
from featurize import *
mnist = np.load("../data/mnist-data.npz")
spam = np.load("../data/spam-data.npz")
toy = np.load("../data/toy-data.npz")


# In[2]:


SEED = 5
np.random.default_rng(seed=SEED)


# ## Question 3: Data Partitioning and Evaluation Metrics

# ### Part a) Data Partitioning: 
# Below I create functions to shuffle, vectorize, and partition the data for use in our model later on. Below are the defined functions

# In[3]:


def vectorize_features(data):
    n = data.shape[0]
    data = data.reshape(n, -1)
    return data

def vectorize_labels(data):
    n = data.shape[0]
    data = data.reshape(-1)
    return data

def shuffle(npz):
    x = npz['training_data']
    y = npz['training_labels']
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    x = vectorize_features(x)
    y = vectorize_labels(y)
    return x, y

def split(features, labels, validation_number):
    train_features = features[validation_number:]
    validation_features = features[:validation_number]
    train_labels = labels[validation_number:]
    validation_labels = labels[:validation_number]
    return train_features, validation_features, train_labels, validation_labels


# And here are the functions implemented on the data

# In[4]:


mnist_x, mnist_y = shuffle(mnist)
train_mnist_f, val_mnist_f, train_mnist_l, val_mnist_l = split(mnist_x, mnist_y, 10000)

spam_x, spam_y = shuffle(spam)
num = round(spam_x.shape[0]*0.20)
train_spam_f, val_spam_f, train_spam_l, val_spam_l = split(spam_x, spam_y, num)


# ### Part b) Evaluation metric:
# A function to evaluate model accuracy

# In[5]:


def compute_s(labels, predictions):
    n = len(labels)
    s = float(1/n) * sum(labels == predictions)
    return s


# ## Question 4: Support Vector Machines

# ### Part a and b) 
# Both MNIST and Spam data sets are run through a train and validation function that trains and evaluates SVM models for a given number of training points. Multiple iterations are conducted to test different number of training points. The entire set of validation points are used each iteration. 

# In[6]:


mnst_model = svm.SVC(kernel='linear') 
spm_model = svm.SVC(kernel='linear')

mnst_num_examples = [100, 200, 500, 1000, 2000, 5000, 10000]
spm_num_examples = [100, 200, 500, 1000, 2000, len(train_spam_f)]

mnst_train_accuracy, mnst_val_accuracy = [], []
spm_train_accuracy, spm_val_accuracy = [], []

def train_and_validate(train_f, train_l, val_f, val_l, model, num):
    x = train_f[:num]
    y = train_l[:num]
    x_val = val_f
    y_val = val_l
    model.fit(x, y)
    y_hat_train = model.predict(x)
    y_hat_val = model.predict(x_val)
    return [compute_s(y, y_hat_train), compute_s(y_val, y_hat_val)]

for num in mnst_num_examples:
    accuracy = train_and_validate(train_mnist_f, train_mnist_l, val_mnist_f, val_mnist_l, mnst_model, num)
    mnst_train_accuracy.append(accuracy[0])
    mnst_val_accuracy.append(accuracy[1])
    
for num in spm_num_examples:
    accuracy = train_and_validate(train_spam_f, train_spam_l, val_spam_f, val_spam_l, spm_model, num)
    spm_train_accuracy.append(accuracy[0])
    spm_val_accuracy.append(accuracy[1])


# ##### The above test is represented visually by the plots below

# In[7]:


plt.title('MNIST Accuracy vs. Number of Training Examples')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.plot(mnst_num_examples, mnst_train_accuracy, label='Training Accuracy')
plt.plot(mnst_num_examples, mnst_val_accuracy, label='Validation Accuracy')
plt.legend()
plt.show()

plt.title('SPAM Accuracy vs. Number of Training Examples')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.plot(spm_num_examples, spm_train_accuracy, label='Training Accuracy')
plt.plot(spm_num_examples, spm_val_accuracy, label='Validation Accuracy')
plt.legend()
plt.show()


# ## Question 5: Hyperparameter Tuning
# Below we conduct hyperparameter training by varying the C regularization parameter in the svm model

# In[8]:


j = 10
k = 10
C_values_mnist = [j**(-7), j**(-6), j**(-5), j**(-4), j**(-3), j**(-2), j**(-1), j**0, j**1]
C_values_spam = [k**(-7), k**(-6), k**(-5), k**(-4), k**(-3), k**(-2), k**(-1), k**0, k**1]

training_accuracy_mnist, validation_accuracy_mnist = [], []

for c in C_values_mnist:
    mnst_model = svm.SVC(C=c, kernel='linear')
    accuracy = train_and_validate(train_mnist_f, train_mnist_l, val_mnist_f, val_mnist_l, mnst_model, 10000)
    training_accuracy_mnist.append(accuracy[0])
    validation_accuracy_mnist.append(accuracy[1])
    


# In[9]:


best_C_mnist = C_values_mnist[np.argmax(validation_accuracy_mnist)]
print(best_C_mnist)


# ##### The best C value is :  1e-06

# ##### The C values used for MNIST testing are  [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# ##### The respective accuracies are: 
# ##### Training = [0.9363, 0.9653, 0.9929, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  
# ##### Validation = [0.9246000000000001, 0.9289000000000001, 0.9159, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072]

# ## Question 6: K-Fold Cross-Validation

# Functions for producing indices and accuracies for one iteration of k-fold are below. 

# In[10]:


def indices(length, k):
    gap = length // k
    indices = []
    for i in range(k+1):
        indices.append(i*k)
    return indices


# In[11]:


def k_fold_iteration(i, j, x, y, model):
    train_x = np.concatenate((x[:i], x[j:]))        
    train_y = np.concatenate((y[:i], y[j:]))
    val_x = x[i:j]
    val_y = y[i:j]
    return train_and_validate(train_x, train_y, val_x, val_y, model, len(train_x))


# 5-fold cross validation is conducted using 5 iterations. After k-fold is completed, accuracies are averaged for each C to determine the most accurate hyperparameter 

# In[12]:


training_accuracy_spam, validation_accuracy_spam = [], []
index_list = indices(len(spam_x), 5)

for c in C_values_spam:
    spm_model = svm.SVC(C=c, kernel='linear')
    temp_train_acc, temp_val_acc = [], []
    for k in range(len(index_list)-1):
        i = index_list[k]
        j = index_list[k+1]
        accuracy = k_fold_iteration(i, j, spam_x, spam_y, spm_model)
        temp_train_acc.append(accuracy[0])
        temp_val_acc.append(accuracy[1])
        
    training_accuracy_spam.append(np.mean(temp_train_acc))
    validation_accuracy_spam.append(np.mean(temp_val_acc))


# In[13]:


best_C_spam = C_values_spam[np.argmax(validation_accuracy_spam)]
print(best_C_spam)


# ##### The best C value is:  10

# ##### The C values used for SPAM testing are  [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2]
# ##### Respective Training accuracies:  [0.712, 0.716, 0.745, 0.765, 0.782, 0.809, 0.836, 0.867, 0.881]  
# ##### Respective Validation sccuracies:  [0.600, 0.64, 0.6, 0.6, 0.72, 0.76, 0.76, 0.8, 0.8]
# ##### The best C value for MNIST and SPAM are 1e-06 and 10 respectively

# ## Question 7: Kaggle

# Now with the best hyperparameters I train and finetune the final pair of SVM MNIST and Spam models.

# In[14]:


mnst_model = svm.SVC(C=best_C_mnist, kernel='linear') 
spm_model = svm.SVC(C=best_C_spam, kernel='rbf')
mnst_model.fit(mnist_x, mnist_y)
spm_model.fit(spam_x, spam_y)
mnist_predictions = mnst_model.predict(vectorize_features(mnist['test_data']))
spam_predictions = spm_model.predict(vectorize_features(spam['test_data']))


# In[15]:


mnist_csv = results_to_csv(mnist_predictions)


# In[16]:


spam_csv = results_to_csv(spam_predictions)


# ##### MNIST achieved 94% accuracy first try. 
# ##### SPAM resulted in a 82% accuracy after playing around with features. 
# I analyzed multiple documents from the spam and ham sets and observed words I felt were frequent across one category and not the other, for example medication, sex, deal. I initially added about 20 words. I noticed during training and validation tests using the features, adding >10 features resulted in a downward trend of the validation indicating some type of overfitting, so I played around with removing some features I originally added. I also removed some of the features already provided until I got a better ratio of features, such that the validation trended towards up as I tested spam with larger amounts of training data. I attempted to add a text length feature, but this made the code time inefficient. I ended up taking out the text length feature. All features edits were made in the featurize.py file, and a screenshot is provided below showing added and omitted features of the final draft. 
