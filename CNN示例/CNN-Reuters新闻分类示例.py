# coding:utf-8
"""
基于keras的Reuters新闻文本分类
"""
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

# 没有涉及特征选择和特征提取称为max_features是不合适的改为vocabulary_size，这里就是按词语频次保留了5000个词
vocabulary_size = 5000
# 文本序列所含最大词语数
maxlen = 400

# 为了显示文本的具体单词内容而加载"词语-索引"词典
word_index = reuters.get_word_index(path="reuters_word_index.json")
index_word = dict()
for word, index in word_index.items():
    if index in index_word:
        print("repeated index: ", index)
    else:
        index_word[index] = word


def main():
    # 读入文本集，除了path和num_words其他参数都用的load_data方法的默认值
    (x_train_seq, y_train), (x_test_seq, y_test) = reuters.load_data(path="reuters.npz",
                                                                     num_words=vocabulary_size,
                                                                     skip_top=0,
                                                                     maxlen=None,
                                                                     seed=113,
                                                                     start_char=1,
                                                                     oov_char=2,
                                                                     index_from=3)

    # 输出给类别下文本数量
    category_text_distribution(y_train)

    # 打印第1篇文本内容
    print("第1篇文本内容\n")
    display_text_content(x_train_seq[0])
    print("\n\n")

    # 截长补短让所有“让每篇文本长度为1000个单词
    x_train = pad_sequences(x_train_seq, maxlen=maxlen)
    x_test = pad_sequences(x_test_seq, maxlen=maxlen)

    # 类别One-Hot编码
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # 输出数据形状
    print("x_train shape = ", x_train.shape)
    print("y_train shape = ", y_train.shape)
    print("x_test shape=", x_test.shape)
    print("y_test shape=", y_test.shape)


    # 构造训练模型
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size,         # 输入单词维度
                        output_dim=300,                    # 输出300维的词向量
                        input_length=maxlen,               # 每篇文本的序列长度
                        trainable=True))
    model.add(Conv1D(filters=256,           # 卷积核数量，卷积层输出维度
                     kernel_size=3,         # 卷积核宽度5个单词
                     activation='relu',     # 激活函数
                     ))
    model.add(MaxPooling1D(pool_size=2))    # 池化窗口大小2个单词
    model.add(Dropout(0.5))                 # 防止卷积层过拟合
    model.add(Flatten())                    # 压平以送入逻辑回归层
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.summary()

    # 设置损失函数、优化器、评测方法
    model.compile(loss='binary_crossentropy',  # 二元交叉熵损失函数
                  optimizer='Adagrad',         # 优化器为Adagrad
                  metrics=['accuracy'])

    # 训练
    model.fit(x_train, y_train,
              batch_size=1000,       # 1个训练周期中，mini-batch gradient descent中每一批次100项数据
              epochs=10,             # 整个数据集将被周期地训练10次，取最优的一次
              verbose=2,             # 显示训练过程
              validation_split=0.2)  # 20%验证数据

    # 测试
    loss, precision = model.evaluate(x_test, y_test, verbose=1)
    print("precision=", precision)


# 统计各类别下文本数量
def category_text_distribution(y_labels):
    category_text_dict = dict()
    for y_label in y_labels:
        category_text_dict.setdefault(y_label, 0)
        category_text_dict[y_label] += 1
    for y_label, cnt in category_text_dict.items():
        print("class %s\t%d"% (y_label, cnt))


# 显示文本内容，主要为了演示语料加载函数load_data中的index_from参数
def display_text_content(word_index_li):
    for index in word_index_li:
        if index >= 3:
            print(index_word[index-3], end=' ')
        else:
            pass
    print()


if __name__ == "__main__":
    main()
