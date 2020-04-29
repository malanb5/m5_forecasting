import tensorflow as tf
from tensorflow import keras
import numpy as np
import traceback
import pandas as pd
from src import Shaper, Timer
from sklearn import metrics, model_selection

class NNPredict:

    @staticmethod
    def _convert_to_score_from_pred(predict_mat):
        """
        converts a prediction array to a composite score to get an estimate of
        the prediction score
        :return: (float) a prediction of the number of sales
        """
        score_pred = list()
        max_score = len(predict_mat[0])
        scores = np.arange(max_score)

        for prob in (predict_mat):
            score = scores.dot(prob.T)
            score_pred.append(score)
        return score_pred

    @staticmethod
    def _make_model(n_inputs, n_output, fan_in):
        model = keras.Sequential([
            keras.layers.Dense(fan_in * 2, input_shape=(n_inputs,)),
            keras.layers.Dropout(rate=.2),
            keras.layers.Dense(fan_in * 3, activation='relu'),
            keras.layers.Dropout(rate=.2),
            keras.layers.Dense(fan_in * 2, activation='relu'),
            keras.layers.Dense(n_output),
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])

        return model

    @staticmethod
    def _train_predict(train_X, train_labels, test_X, test_labels, n_outputs, n_inputs, epochs=20, verbosity=1,
                       evaluate=0, fan_in = 4):

        model = NNPredict._make_model( n_inputs, n_outputs, fan_in=fan_in)
        model.fit(train_X, train_labels, epochs=epochs)
        model.save('nnpredict_20_4_20.hdf5')

        # model = keras.models.load_model("nnpredict_20_4_20.hdf5")
        # test_loss, test_acc = model.evaluate(train_X, train_labels, verbose=verbosity)
        #
        # if verbosity:
        #     print("\nTest Accuracy:", test_acc)

        probability_model = tf.keras.Sequential([model,
                                                 tf.keras.layers.Softmax()])

        predictions = probability_model.predict(test_X)

        scores = NNPredict._convert_to_score_from_pred(predictions)

        if evaluate:
            loss = metrics.mean_squared_error(scores, test_labels)

            if verbosity:
                print("The loss for the test set: %f" %loss)

        elif evaluate == 0:
            return scores

    @staticmethod
    def convert_to_np(df):
        if isinstance(df, pd.Series) or isinstance(df, pd.DataFrame):
            return np.asarray(df)
        else:
            print("is not dataframe or series")

    @staticmethod
    def exec(device_type, y_train, X_train, test_proportion, seed, shuffle, X_test, verbosity, epochs, fan_in=4):

        y_train = NNPredict.convert_to_np(y_train)
        X_train = NNPredict.convert_to_np(X_train)
        X_test = NNPredict.convert_to_np(X_test)

        n_inputs = len(X_train[0])

        if len(y_train.shape) == 1:
            n_outputs = 1
        else:
            n_outputs = y_train.shape[1]

        # do not split to make predictions
        if test_proportion > 0:
            X_train, X_test, y_train, y_test = \
                model_selection.train_test_split(X_train, y_train, test_size=test_proportion, random_state=seed, shuffle=shuffle)
        elif test_proportion == 0:
            y_test = []
        elif test_proportion < 0:
            raise Exception("Cannot have a negative test size.")



        if verbosity:
            tf.config.set_soft_device_placement(True)
            tf.debugging.set_log_device_placement(True)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        cpus = tf.config.experimental.list_physical_devices('CPU')

        scores = list()

        if gpus and device_type == "GPU":
            try:
                for gpu in gpus:
                    # made and train the model on the GPU
                    with tf.device(gpu.name.replace('physical_device:', '')):
                        scores = NNPredict._train_predict(X_train, y_train, X_test, y_test, n_outputs,
                                                          n_inputs, epochs, verbosity, evaluate=test_proportion, fan_in=fan_in)


            except RuntimeError as e:
                print(e)
                traceback.print_tb(e)
                traceback.print_stack(e)
        elif device_type == "CPU":
            for cpu in cpus:
                with tf.device(cpu.name.replace('physical_device:', "")):
                    scores = NNPredict._train_predict(X_train, X_test, y_train, y_test, n_outputs, n_inputs,
                                                      epochs, verbosity, evaluate=test_proportion)


        if test_proportion == 0:
            print("scores: %s" %str(scores))
            print("size of scores: %d" %(len(scores)))
            Shaper.save(scores, "eda_objs/scores_nnpredict_np_array_%s.pkl" % Timer.get_timestamp_str())

    @staticmethod
    def make_predictions(device_type):
        epochs = 1
        verbosity = 1
        test_size = 0
        seed =7775
        shuffle = 1

        X_train = Shaper.load("objs/X_train.pkl")
        y_train = Shaper.load("objs/y_train.pkl")

        X_test = Shaper.load("objs/days/predict_0.pkl")

        print(X_test)

        NNPredict.exec(device_type, y_train, X_train, test_size, seed, shuffle, X_test, verbosity, epochs, fan_in =4)


devices = [ "GPU"]
for device in devices:
    Timer.time_this(NNPredict.make_predictions, **{"device_type": device})
