# coding:utf-8
"""
基于keras的IMDB影评情感分类
"""

# 导入数字列表类
from keras.preprocessing import sequence
# 导入线性堆叠模型类
from keras.models import Sequential
# 导入嵌入层类
from keras.layers.embeddings import Embedding
# 导入Dropout层类
from keras.layers.core import Dropout
# 导入平坦层类
from keras.layers.core import Flatten
# 导入隐藏层类
from keras.layers.core import Dense
# 导入1维卷积类
from keras.layers import Conv1D
# 导入池化层
from keras.layers import MaxPooling1D, GlobalMaxPooling1D
# 导入数据集
from keras.datasets import imdb


# 最大特征词语数量
max_features = 5000
# 文本序列所含最大词语数
maxlen = 400
# 嵌入层输出维度，也就是每个词语向量的维度
embedding_dims = 50
# 卷积核数量，也就是卷积层的输出维度
filters = 250
# 卷积核宽度，3个词语
kernel_size = 3
# 隐藏层输出维度
hidden_dims = 250
# 训练1个批次使用的样本数
batch_size = 32
# 训练阶段遍历1遍全部训练集叫做1个epoch，这里定义的是周期数
epochs = 10


def main():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
        num_words=max_features,  # 最大可能的词语索引为num_words-1
        skip_top=0,
        maxlen=None,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    # 截长补短让所有“让每篇文本长度为maxlen个单词
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    # 构建神经网络模型
    model = Sequential()

    # 嵌入层，将文本集进行向量化
    vocabulary_size = max_features                      # num_words参数的值等于嵌入层词汇表长度
    model.add(Embedding(output_dim=embedding_dims,      # 嵌入层输出维度为300
                        input_dim=vocabulary_size,      # 嵌入层矩阵行数（词语数量），此处要填好大小，否则训练时会出现越界错误
                        input_length=maxlen,            # 1篇文本由100个词语构成
                        trainable=True))                # 该参数是Embedding类的基类成员，默认为True，层权重在训练过程中更新
    # 加入Dropout层防止嵌入层过拟合
    model.add(Dropout(0.2))

    # 加入卷积层
    model.add(Conv1D(filters=filters,                   # 64个卷积核，卷积层输出的维度为64
                     kernel_size=kernel_size,           # 卷积核宽度为3个词语，一般选择2～3个词语
                     padding='same',                    # 卷积运算后仍然保持句子长度不变
                     activation='relu'))                # 只保留正值特征
    # 加入池化层方法1
    model.add(GlobalMaxPooling1D())                     # 对每一个卷积输出的列向量求最大值，减掉1维，因此后边不需要Flatten层了

    # # 加入平坦层，以使得可以输出到后边输出层
    # model.add(Flatten())

    # 隐藏层
    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.2))

    # 输出层（多类别时为逻辑回归）
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # 设置模型
    model.compile(loss='binary_crossentropy',  # 损失函数
                  optimizer='adam',            # 优化器
                  metrics=['accuracy'])        # 模型评测方法

    # 打印模型摘要
    model.summary()

    # 训练
    model.fit(x_train, y_train,
            batch_size=batch_size,  # 每一批次100项数据
            epochs=epochs,          # 执行10个训练周期
            verbose=2,              # 显示训练过程
            validation_split=0.2)   # 20%的训练数据作为验证数据

    # 评测
    loss, precision = model.evaluate(x_test, y_test, verbose=1)
    print("precision=", precision)


if __name__ == "__main__":
    main()
