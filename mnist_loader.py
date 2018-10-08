#!/bin/env python3
# coding: utf-8

"""
Библиотека загружающая датасет изображений рукописных цифр из базы MNIST.
"""

import cPickle
import gzip

import numpy as np

def load_data():
    """Загружает данные MNIST в форме массива, содержащего обучающую
    выборку, валидационную выборку, и тестовую выборку.

    Переменная ``training_data`` возвращается как массив с двумя запясими.
    Первая запись содержит тренировочные изображения. Это многомерный
    массив numpy размером в 50,000 объектов. Каждый объект, в свою очередь,
    это многомерный массив numpy из 784 значений, представляющих
    28 * 28 = 784 пикселя из одного изображения MNIST.

    Вторая запись в массиве ``training_data`` это многомерный массив numpy,
    содержащий 50,000 записей. Эти записи - значения цифр (от 0 до 9),
    соответствующие изображениям содержащимся в первой записи массива.

    Переменные ``validation_data`` и ``test_data`` устроенны аналогично,
    но содержать только по 10,000 изображений.

    Такой формат данных удобен, но для обучения нейросети будет лучше
    немного изменить формат тренировочной выборки ``training_data``.
    Это выполняет функция ``load_data_wrapper()`` пердставленная ниже.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Возвращает массив содержащий ``(training_data, validation_data,
    test_data)``, обращаюясь к функции load_data(), но формат данных
    более удобен для использования в нашей нейросети.

    В частности, ``training_data`` представляется списком, содержащим
    50,000 двойных массивов ``(x, y)``. ``x`` – 784-мерный numpy массив
    содержащий взодное изображение. ``y`` – 10-мерный numpy массив,
    представляющий единичный вектор, соответствующий корректному
    значению для ``x``.

    Переменные ``validation_data`` и ``test_data`` представленны списками,
    содержащими по 10,000 двойных массивов ``(x,y)``. В каждом случае ``x``
    это 784-мерный nupmy массив, содержащий входное изображение, а ``y``
    – соответствующий класс, то есть цифры соответствующие ``x``.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Возвращает 10-мерный единичный вектор со значение 1.0 на j-ой
    позиции и нулями на всех остальных. Это используется чтобы
    конвертировать цифры (0...9) в соответствующий желаемый вывод
    нейронной сети
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
