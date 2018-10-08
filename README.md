# Нейронная сеть на Python для классификации рукописных цифр.

На Python 3.7 реализована двуслойная нейронная сеть прямого распостранения, обучающаяся с помощью стохастического градиентного спуска. Вычисление градиентов выполняется алгоритмом обратного распостранения ошибки. В качестве функции активации используется сигмоида.

Код нейронной сети находится в файле [neuralnet.py](https://github.com/antonprokopovich/neuralnet/blob/master/neuralnet.py). В коде подробно прокомментированы этапы реализации модели, включая весь лежащий в основе математический аппарат.

Обученающая и тестовая выбоки получены из базы данных рукописных цифр MNIST. Функции загружающие данные расположены в файле [mnist_loader.py](https://github.com/antonprokopovich/neuralnet/blob/master/mnist_loader.py).

Файл используется для запуска процесса обучения и последующей оценки качества работы нейросети на обучающей и тестовой выборках соответственно – [test.py](https://github.com/antonprokopovich/neuralnet/blob/master/test.py).
