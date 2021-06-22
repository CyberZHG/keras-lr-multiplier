# Keras LR Multiplier

[![Travis](https://travis-ci.org/CyberZHG/keras-lr-multiplier.svg)](https://travis-ci.org/CyberZHG/keras-lr-multiplier)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-lr-multiplier/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-lr-multiplier)

**This repo is outdated and will no longer be maintained.**

Learning rate multiplier wrapper for optimizers.

## Install

```bash
pip install git+https://github.com/CyberZHG/keras-lr-multiplier.git
```

## Usage

### Basic

`LRMultiplier` is a wrapper for optimizers to assign different learning rates to specific layers (or weights). The first argument is the original optimizer which could be either an identifier (e.g. `'Adam'`) or an initialized object (e.g. `Adam(lr=1e-2)`). The second argument is a dict that maps prefixes to learning rate multipliers. The multiplier for a weight is the value mapped from the __longest matched prefix__ in the given dict, and the default multiplier `1.0` will be used if there is no prefix matched.

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

### Lambda

The multiplier can be a callable object. The input argument is the number of steps starting from 0.

```python
from keras import backend as K
from keras_lr_multiplier import LRMultiplier

LRMultiplier('adam', {'Dense': lambda t: 2.0 - K.minimum(1.9, t * 1e-4)})
```
