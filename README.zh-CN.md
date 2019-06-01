# Keras学习率分层倍率控制

[![Travis](https://travis-ci.org/CyberZHG/keras-lr-multiplier.svg)](https://travis-ci.org/CyberZHG/keras-lr-multiplier)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-lr-multiplier/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-lr-multiplier)
[![Version](https://img.shields.io/pypi/v/keras-lr-multiplier.svg)](https://pypi.org/project/keras-lr-multiplier/)
[![996.ICU](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://996.icu) 
![Downloads](https://img.shields.io/pypi/dm/keras-lr-multiplier.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-theano-blue.svg)
![](https://img.shields.io/badge/keras-cntk-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-lr-multiplier/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-lr-multiplier/blob/master/README.md)\]

在Keras中控制每一层的学习率，通过层名的最长匹配前缀来确定学习率倍率，适用于按Keras规则写的优化器。

## 安装

```bash
pip install keras-lr-multiplier
```

## 使用

### 基本

`LRMultiplier`中第一个参数是原有的优化器，形式和`compile`相同，可以是标志字符串（如`'Adam'`），也可以是一个初始化后的优化器（如`Adam(lr=1e-2)`）。第二个参数是一个`dict`，是名称前缀到学习率倍率的映射，每一层在没有匹配到任何前缀的情况下默认倍率取`1.0`，否则只采用最长匹配前缀的结果，如存在`"Dense"`和`"Dense-1"`时，`"Dense-12"`采用`"Dense-1"`对应的倍率。一个例子如下：

```python
from keras.models import Sequential
from keras.layers import Dense
from keras_lr_multiplier import LRMultiplier

model = Sequential()
model.add(Dense(
    units=5,
    input_shape=(5,),
    activation='tanh',
    name='Dense',
))
model.add(Dense(
    units=2,
    activation='softmax',
    name='Output',
))
model.compile(
    optimizer=LRMultiplier('adam', {'Dense': 0.5, 'Output': 1.5}),
    loss='sparse_categorical_crossentropy',
)
```

### 变化倍率

倍率也可以是一个可以被调用的对象，如匿名函数或实现了`__call__`的`object`。输入的参数是从`0`开始的训练步数，需要返回对应的学习率倍率：

```python
from keras import backend as K
from keras_lr_multiplier import LRMultiplier

LRMultiplier('adam', {'Dense': lambda t: 2.0 - K.minimum(1.9, t * 1e-4)})
```
