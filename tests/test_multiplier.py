import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_lr_multiplier.backend import models, layers, optimizers, callbacks
from keras_lr_multiplier import LRMultiplier


class TestMultiplier(TestCase):

    def test_compare_rate(self):
        inputs = np.random.standard_normal((1024, 5))
        outputs = (inputs.dot(np.random.standard_normal((5, 1))).squeeze(axis=-1) > 0).astype('int32')
        weight = np.random.standard_normal((5, 2))

        model = models.Sequential()
        model.add(layers.Dense(
            units=2,
            input_shape=(5,),
            use_bias=False,
            activation='softmax',
            weights=[weight],
            name='Output',
        ))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.fit(inputs, outputs, epochs=30)
        default_loss = model.evaluate(inputs, outputs)

        model = models.Sequential()
        model.add(layers.Dense(
            units=2,
            input_shape=(5,),
            use_bias=False,
            activation='softmax',
            weights=[weight],
            name='Output',
        ))
        model.compile(optimizer=LRMultiplier('adam', {'Output': 2.0}), loss='sparse_categorical_crossentropy')
        model.fit(inputs, outputs, epochs=30)
        model_path = os.path.join(tempfile.gettempdir(), 'test_lr_multiplier_%f.h5' % np.random.random())
        model.save(model_path)
        model = models.load_model(model_path, custom_objects={'LRMultiplier': LRMultiplier})
        quick_loss = model.evaluate(inputs, outputs)
        self.assertLess(quick_loss, default_loss)

        predicted = model.predict(inputs).argmax(axis=-1)
        self.assertLess(np.sum(np.abs(outputs - predicted)), 300)

    def test_lr_plateau(self):
        inputs = np.random.standard_normal((1024, 5))
        outputs = (inputs.dot(np.random.standard_normal((5, 1))).squeeze(axis=-1) > 0).astype('int32')

        model = models.Sequential()
        model.add(layers.Dense(
            units=2,
            input_shape=(5,),
            use_bias=False,
            activation='softmax',
            name='Output',
        ))
        model.compile(
            optimizer=LRMultiplier(optimizers.Adam(), {'Output': 100.0}),
            loss='sparse_categorical_crossentropy',
        )
        model.fit(
            inputs,
            outputs,
            validation_split=0.1,
            epochs=1000,
            callbacks=[
                callbacks.ReduceLROnPlateau(patience=2, verbose=True),
                callbacks.EarlyStopping(patience=5),
            ],
        )

        predicted = model.predict(inputs).argmax(axis=-1)
        self.assertLess(np.sum(np.abs(outputs - predicted)), 20)

    def test_restore_weights(self):
        inputs = np.random.standard_normal((1024, 5))
        outputs = (inputs.dot(np.random.standard_normal((5, 1))).squeeze(axis=-1) > 0).astype('int32')
        weight = np.random.standard_normal((5, 2))

        model = models.Sequential()
        model.add(layers.Dense(
            units=2,
            input_shape=(5,),
            use_bias=False,
            activation='softmax',
            weights=[weight],
            name='Output',
        ))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.fit(inputs, outputs, shuffle=False, epochs=30)
        one_pass_loss = model.evaluate(inputs, outputs)

        model = models.Sequential()
        model.add(layers.Dense(
            units=2,
            input_shape=(5,),
            use_bias=False,
            activation='softmax',
            weights=[weight],
            name='Output',
        ))
        model.compile(optimizer=LRMultiplier('adam', {}), loss='sparse_categorical_crossentropy')
        model.fit(inputs, outputs, shuffle=False, epochs=15)
        model_path = os.path.join(tempfile.gettempdir(), 'test_lr_multiplier_%f.h5' % np.random.random())
        model.save(model_path)
        model = models.load_model(model_path, custom_objects={'LRMultiplier': LRMultiplier})
        model.fit(inputs, outputs, shuffle=False, epochs=15)
        two_pass_loss = model.evaluate(inputs, outputs)
        self.assertAlmostEqual(one_pass_loss, two_pass_loss, places=2)

    def test_repeated_weights(self):
        inputs = np.random.standard_normal((1024, 5))
        outputs = (inputs.dot(np.random.standard_normal((5, 1))).squeeze(axis=-1) > 0).astype('int32')

        model = models.Sequential()
        model.add(layers.Dense(
            units=5,
            input_shape=(5,),
            activation='tanh',
            name='Dense',
        ))
        model.add(layers.Dense(
            units=2,
            activation='softmax',
            name='Output',
        ))
        model.compile(
            optimizer=LRMultiplier('adam', {'Dense': 0.5, 'Output': 1.5}),
            loss='sparse_categorical_crossentropy',
        )
        model.fit(
            inputs,
            outputs,
            validation_split=0.1,
            epochs=1000,
            callbacks=[
                callbacks.ReduceLROnPlateau(patience=2, verbose=True),
                callbacks.EarlyStopping(patience=5),
            ],
        )
        predicted = model.predict(inputs).argmax(axis=-1)
        self.assertLess(np.sum(np.abs(outputs - predicted)), 20)
