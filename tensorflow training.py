import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


os.chdir('/Users/arunkarthik/Downloads')

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)

le = LabelEncoder()
ohe = OneHotEncoder()

data = pd.read_csv('master.csv')

data.drop(['HDI for year','country-year'],axis=1,inplace=True)

data.rename(columns={'suicides/100k pop':'suicides/100k',' gdp_for_year ($) ':'gdp_for_year','gdp_per_capita ($)':'gdp_per_capita'},inplace=True)


data['gdp_for_year'] = data['gdp_for_year'].apply(lambda x:x.replace(',','')).astype(int)

data['gdp_for_year'] = (data['gdp_for_year']/1000000)

X = data.drop(['generation'],axis='columns')
y = data['generation']



X['country'] = le.fit_transform(X['country'])
X['age'] = le.fit_transform(X['age'])
X['sex'] = le.fit_transform(X['sex'])


print(X.head())

y = pd.get_dummies(y)
print(y.head())
y = y.drop(y['Silent'],axis='columns',inplace=True)
y = np.array(y)

learning_rate = 0.1
epoch = 100
classes = 5
n_dim = X.shape[1]
cost_hist = np.empty(shape=[1],dtype=float)

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=343,test_size=0.3)
print(x_train.shape,'xtrain')
print(y_train.shape,'ytrain')
print(x_test.shape,'xtest')
print(y_test.shape,'ytest')

hidden_1  = 100
hidden_2 = 100
hidden_3 = 100
hidden_4 = 100

x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,classes]))
b = tf.Variable(tf.zeros([classes]))
y_ = tf.placeholder(tf.float32,[None,classes])

def multi_layer_precp(x,weigth,bias):
    layer_1 = tf.add(tf.matmul(x,weigth['w1']),bias['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weigth['w2']),bias['b2'])
    layer_2 = tf.nn.softmax(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,weigth['w3']),bias['b3'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3,weigth['w4']),bias['b4'])
    layer_4 = tf.nn.softmax(layer_4)

    out_layer = tf.matmul(layer_4,weigth['out'])+bias['out']
    return out_layer


weight = {
    'w1' : tf.Variable(tf.truncated_normal([n_dim,hidden_1])),
    'w2' : tf.Variable(tf.truncated_normal([hidden_1,hidden_2])),
    'w3' : tf.Variable(tf.truncated_normal([hidden_2,hidden_3])),
    'w4' : tf.Variable(tf.truncated_normal([hidden_3,hidden_4])),
    'out': tf.Variable(tf.truncated_normal([hidden_4,classes]))

}

bias = {
    'b1' : tf.Variable(tf.truncated_normal([hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([hidden_4])),
    'out': tf.Variable(tf.truncated_normal([classes]))

}


sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

y1 = multi_layer_precp(x,weight,bias)
cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y1,labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_fun)


accuracy_hist = []
mse_hist=[]

for epochs in range(epoch):
    sess.run(training_step,feed_dict={x:x_train,y_:y_train})
    cost = sess.run(cost_fun,feed_dict={x:x_train,y_:y_train})
    cost_hist = np.append(cost_hist,cost)
    correct_pred = tf.equal(tf.argmax(y1,1),tf.argmax(y_,1))
    accuracy  = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    y_pred = sess.run(y1,feed_dict={x:x_test})
    mse = tf.reduce_mean(tf.square(y_pred-y_test))
    mse_ = sess.run(mse)
    accuracy = sess.run(accuracy,feed_dict={x:x_train,y_:y_train})
    accuracy_hist.append(accuracy)

    print("epoch: ", epoch, '--', 'cost:', cost, '-MSE: ', mse_, 'training accuracy: ', accuracy)

plt.plot(mse_hist, 'r')
plt.show()
plt.plot(accuracy_hist)
plt.show()

correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y1, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: x_train, y_: y_train}))

y_pred = sess.run(y1, feed_dict={x: x_test})
print("MSE %.4F" % sess.run(mse))


