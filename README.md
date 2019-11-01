# 神经网络学习笔记

手敲《深度学习Keras快速开发入门》乐毅2017年6月书中各章代码。之所以手敲一遍，一方面熟悉Keras开发环境，另一方面熟悉基本的神经网络模型，最重要的是书似乎没配套代码。这本书的前八章介绍了Keras的各个方面的使用方法，第9章通过构建一个多类别文本分类模型将前八章内容串了起来。

<!-- TOC -->

- [神经网络学习笔记](#神经网络学习笔记)
    - [1. Keras](#1-keras)
        - [第3章 Keras快速上手](#第3章-keras快速上手)
        - [第4章 Keras模型的定义](#第4章-keras模型的定义)
        - [第5章 Keras网络结构](#第5章-keras网络结构)
        - [第9章 Keras实战](#第9章-keras实战)
    - [2. Theano](#2-theano)

<!-- /TOC -->

## 1. Keras

### 第3章 Keras快速上手

本章初步认识Keras的一种模型Sequential，这个模型可以翻译为堆叠模型。我们都知道神经网络的基本单位是神经元，而一个或多个神经元可以构成一个网络层，多个网络层就构成了一个神经网络模型。那么堆叠模型就是让用户不断向其中堆叠网络层进而构成神经网络。本章的例题演示了一个基本的MLP神经网络构成的多类别分类模型。[Keras快速上手笔记](https://github.com/baiziyuandyufei/start_keras/blob/master/Chap3/Chap3%20Keras%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B.ipynb)

### 第4章 Keras模型的定义

Keras有两种类型的模型：堆叠模型（Sequential）和函数式模型（Model)。Sequential模型是函数式模型的一种特殊情况。本节构建一个MLP神经网络解决两类别分类问题。通过该示例，演示Sequential模型。
[实践笔记](https://github.com/baiziyuandyufei/start_keras/blob/master/Chap4/Chap4%20Keras%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%AE%9A%E4%B9%89.ipynb)
[文本笔记](https://zhuanlan.zhihu.com/p/79883773)

### 第5章 Keras网络结构

Keras由各种网络层构成，Keras的网络结构就是指各种网络层，本章介绍Keras中提供的各种网络层。
[文本笔记1](https://github.com/baiziyuandyufei/start_keras/blob/master/Chap5/Chap5%20Keras%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.md)
[文本笔记2](https://zhuanlan.zhihu.com/p/79883773)

### 第9章 Keras实战

本章介绍如何使用Keras构建一个简单的CNN神经网络，通过20newsgroup新闻语料训练一个20类别的分类器，并在验证集上验证了该模型的准确率。详细代码可以参考[using pre trained word embeddings in a keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html?spm=a2c4e.11153959.blogcont221681.21.4ec062f0OhwRQk)

## 2. Theano
