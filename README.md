# 深度学习Keras快速开发入门

手敲《深度学习Keras快速开发入门》乐毅2017年6月书中各章代码。之所以手敲一遍，一方面熟悉Keras开发环境，另一方面熟悉基本的神经网络模型，最重要的是书似乎没配套代码。这本书的前八章介绍了Keras的各个方面的使用方法，第9章通过构建一个多类别文本分类模型将前八章内容串了起来。

<!-- TOC -->

- [深度学习Keras快速开发入门](#深度学习keras快速开发入门)
    - [第3章 Keras快速上手](#第3章-keras快速上手)
    - [第4章 Keras模型的定义](#第4章-keras模型的定义)
    - [第5章 Keras网络结构](#第5章-keras网络结构)
    - [第6章 Keras数据预处理](#第6章-keras数据预处理)
    - [第7章 Keras内置网络配置](#第7章-keras内置网络配置)
    - [第8章 Keras实用技巧和可视化](#第8章-keras实用技巧和可视化)
    - [第9章 Keras实战](#第9章-keras实战)

<!-- /TOC -->

## 第3章 Keras快速上手

本章初步认识Keras的一种模型Sequential，这个模型可以翻译为堆叠模型。我们都知道神经网络的基本单位是神经元，而一个或多个神经元可以构成一个网络层，多个网络层就构成了一个神经网络模型。那么堆叠模型就是让用户不断向其中堆叠网络层进而构成神经网络。本章的例题演示了一个基本的MLP神经网络构成的多类别分类模型。

## 第4章 Keras模型的定义

Keras有两种类型的模型：堆叠模型（Sequential）和函数式模型（Model)。Sequential模型是函数式模型的一种特殊情况。本节构建一个MLP神经网络解决两类别分类问题。通过该示例，演示Sequential模型。

## 第5章 Keras网络结构

Keras由各种网络层构成，Keras的网络结构就是指各种网络层，本章介绍Keras中提供的各种网络层。

## 第6章 Keras数据预处理

## 第7章 Keras内置网络配置

## 第8章 Keras实用技巧和可视化

本章介绍Keras的调试方法以及模型可视化方法。

## 第9章 Keras实战

本章介绍如何使用Keras构建一个简单的CNN神经网络，通过20newsgroup新闻语料训练一个20类别的分类器，并在验证集上验证了该模型的准确率。详细代码可以参考[using pre trained word embeddings in a keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html?spm=a2c4e.11153959.blogcont221681.21.4ec062f0OhwRQk)
