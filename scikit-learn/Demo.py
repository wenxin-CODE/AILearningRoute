import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers


tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')

batchsz = 128

#加载数据及数据预处理
# the most frequest words
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# x_train:[b, 80]
# x_test: [b, 80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)



class MyRNN(keras.Model):
    #定义层的实现
    def __init__(self, units):
        super(MyRNN, self).__init__()


        # 将词转为词向量（total_words：单词表大小, embedding_len：词向量长度，input_length：句子的最大长度）
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)


        # [b, 80, 100]，units:输出空间的维度(正整数),即隐藏层神经元数量(这里是64)
        self.rnn = keras.Sequential([
            # #创建双层RNN
            # layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            # layers.SimpleRNN(units, dropout=0.5, unroll=True)

            #创建双层LSTM
            #unroll：默认为False，为True可以加速RNN（占用大量内存），为True仅适用于短序列。
            layers.LSTM(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.LSTM(units, dropout=0.5, unroll=True)

            # #创建双层GRU
            # layers.GRU(units, dropout=0.5, return_sequences=True, unroll=True),
            # layers.GRU(units, dropout=0.5, unroll=True)
        ])


        # fc全连接层,用于分类,[b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    # 实现前向过程
    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 80, 100] => [b, 64]
        x = self.rnn(x)

        # out: [b, 64] => [b, 1]
        x = self.outlayer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob

#训练模型
def main():
    units = 64
    epochs = 4

    import time

    t0 = time.time()

    # MyRNN：网络的实例化，compile：网络的装载，fit：网络的训练，evaluate：网络的测试
    model = MyRNN(units)
    model.compile(optimizer = keras.optimizers.Adam(0.001),
                  loss = tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)


    t1 = time.time()
    print('total time cost:', t1-t0)


if __name__ == '__main__':
    main()
