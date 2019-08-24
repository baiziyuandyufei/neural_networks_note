# Chap5 Keras网络结构

## 1. Dense层

Dense是常用的全连接层，所实现的运算是output=activation(dot(input, kernel) + bias)。其中，activation是逐元素计算的激活函数；kernel是本层的权值矩阵；bias为偏置向量，只有当user_bias=True时，bias设置才生效。

### 参数

- units是一个整数，指定Dense层输出的维度

- input_shape是一个整数元组，描述输入矩阵的形状，需要注意的是当Dense层为模型首层时，input_shape元组的元素不对输入样本数量进行指定，也就是说input_shape的第1个元素值为输入数据矩阵的列数或称维度input_dim。当Dense为模型首层时，input_shape=(input_dim,)

### 输入

矩阵(nb_samples, input_dim)

### 输出

矩阵(nb_samples, units)

## 2. Dropout层

为输入数据施加Dropout。Dropout将在训练过程中每次更新时，随机断开一定百分比（rate）的输入神经元。Dropout层用于防止过拟合。

### 参数

- rate是一个浮点数，用于控制所需要断开的神经元的比例

## 3. Flatten层

Flatten层用来将输入“压平“，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

Flatten层将一个维度大于或等于3的高维矩阵压扁为一个二维的地位矩阵。其压缩方法是保留第一个维度的大小，然后将所有剩下的数据压缩到第二个维度中，因此第二个维度的大小是原矩阵第二个维度之后所有维度大小的乘积。这里第一个维度通常是每次迭代所需的小批量文本数量，而压缩后的第二个维度就是表达一句文本所需的向量长度。

## *4. Permute层

Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。

## *5. Lambda层

本函数用于对上一层的输出施以任何Theano/Tensorflow表达式。

## 6. Conv1D卷积层

一维卷积通常成为时域卷积，需要注意的是它与信号处理中的一维卷积略有不同，因为信号处理中的信号是时间上的一维信号，而文本中的数据是时间上的多维信号，因为一篇文本或一个句子是由多个词语构成的，且一个词语一般都会有100～300维。

### 参数

- filters表示输出的维度（注意指的是一个词语向量的维度）
- kernel_size表示滤波器长度

### 输入

形如(samples, steps, input_dim)的3维矩阵，显然samples表示文本的数量，steps表示一句文本所含词语数量，input_dim表示一个词语的向量维度

### 输出

形如(samples, new_steps, nb_filter)显然samples依然表示文本的数量且其值不变，new_steps表示一句文本所含新词语数量，nb_filter表示一个词语向量的维度

## 7. *Conv2DTranspose层

该层是转置的卷积操作（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。例如，将具有该卷积层输出shape的tensor还原为具有该卷积层输入shape的tensor。同时保留与卷积层兼容的连接模式。此方法在生成对抗网络（GAN）中可能用到。

## 8. *UpSampling1D层

对1D输入数据的上采样层。在时间轴上，将每个时间步重复length次。

## 9. *ZeroPadding1D层

对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度。

## 10. MaxPolling1D层

对时域1D信号进行最大值池化。池化的目的是为了计算特征在局部的充分统计量，从而降低总体的特征数量，防止过度拟合和减少计算量。

### 参数

- pool_size是一个整数，表示池化窗口大小。

## 11. 循环层

这是循环层的抽象类，继承该类的子类有LSTM、GRU或SimpleRNN。

## 12. LSTM层

Keras的长短期记忆模型。

## 13. 嵌入层

嵌入层一般只放在神经网络模型的第一层，目的是将文本的单词索引序列转换为单词向量序列。

### 输入

形如(samples, sequence_length)的2D矩阵。显然saples为句子数量，sequence_length为一个句子中所含词语数量（句子经过padding）。

### 输出

形如(samples, sequence_length, output_dim)的3维向量。显示samples依然是句子数量其值未变，sequence_length表示一个句子含有的单词数量其值未变，output_dim表示一个词语向量的维度。

### 参数

- input_dim输入文本集对应词典的长度
- output_dim输出后一个单词向量的维度
- input_length一个句子含有的固定单词数量。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。