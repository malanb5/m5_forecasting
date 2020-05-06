from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import Embedding, Add, Input, concatenate, SpatialDropout1D
from keras.layers import Flatten, Dropout, Dense, BatchNormalization, Concatenate
from tqdm.keras import TqdmCallback
import tensorflow as tf
import traceback
from yj import Timer, FManager, Shaper
from yj.runners.BaseRunnerImpl import BaseRunnerImpl
from datetime import timedelta
import pandas as pd, numpy as np
from keras.models import load_model

class NNRunner(BaseRunnerImpl):

    def _make_embedded_layer(self, shape, name, max_val, embed_n):
        input = Input(shape=shape, name=name)
        embed = Embedding(max_val, embed_n)(input)

        return input, embed

    def _make_dense_input_layer(self, shape, name, n_d_layers, act_type):
        input = Input(shape=shape, name=name)
        d_l = Dense(n_d_layers, activation=act_type)(input)
        d_l = BatchNormalization()(d_l)

        return input, d_l

    def _make_keras_input(self, df, cols, X = None):
        if X is None:
            X = {}
            for each_col in cols:
                X[each_col] = df[each_col]
        else:
            for each_col in cols:
                X[each_col] = df[each_col]
        return X

    def train(self, cuda):
        model_fp = 'models/wal_nn_%s.hdf5 '% Timer.get_timestamp_str()

        ds = FManager.load("objs/X_train.pkl")
        y = FManager.load("objs/y_train.pkl")

        cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday',
                    'month', 'year', 'event_name_1', 'event_name_2', 'event_type_1',
                    'event_type_2', 'week', 'quarter', 'mday']
        cont_cols = ['sell_price', 'lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7',
                     'rmean_7_28', 'rmean_28_28']


        input_layers = []
        hid_layers = []

        n_embed_out = 750
        dense_n = 3000
        batch_size = 1000
        epochs = 5
        lr_init, lr_fin = 10e-5, 10e-6

        exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
        steps = int(len(ds) / batch_size) * epochs
        lr_decay = exp_decay(lr_init, lr_fin, steps)

        for cat_col in cat_cols:
            max_cat = Shaper.get_max(ds, cat_col)
            in_layer, embed_layer = self._make_embedded_layer([1], cat_col, max_cat, n_embed_out)
            input_layers.append(in_layer)
            hid_layers.append(embed_layer)

        fe = concatenate(hid_layers)
        s_dout = SpatialDropout1D(0.1)(fe)
        x = Flatten()(s_dout)

        con_layers = []

        for con_col in cont_cols:
            in_layer, embed_layer = self._make_dense_input_layer([1], con_col, n_embed_out, 'relu')
            input_layers.append(in_layer)
            con_layers.append(embed_layer)

        con_fe = concatenate(con_layers)

        x = concatenate([x, con_fe])
        x = Dropout(0.1)(Dense(dense_n, activation='relu')(x))
        x = Dropout(0.1)(Dense(dense_n ,activation='relu')(x))
        x = Dropout(0.1)(Dense(dense_n ,activation='relu')(x))
        x = Dropout(0.1)(Dense(dense_n, activation='relu')(x))
        x = Dropout(0.1)(Dense(dense_n, activation='relu')(x))
        outp = Dense(1, kernel_initializer='normal' ,activation='linear')(x)

        optimizer_adam = Adam(lr=lr_fin, decay=lr_decay)

        model = Model(inputs=input_layers, outputs=outp, name="wal_net")
        model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])
        model.summary()

        X = self._make_keras_input(ds, cat_cols)
        X = self._make_keras_input(ds, cont_cols, X)

        checkpoint_name = 'models/weights-{epoch:03d}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        model.fit(X, y, batch_size=batch_size, use_multiprocessing=True,
                  validation_split=.1, epochs=epochs, shuffle=True, verbose=0,
                  callbacks=[TqdmCallback(verbose=2) ,checkpoint, es])

        model.save(model_fp)

    def predict(self):
        from yj.environment import ALPHAS, WEIGHTS, TR_LAST, MAX_LAGS, FDAY, SUBMISSION_FP, PRICE_CAL_TRAIN_DF_FP,\
            NN_MODEL_FP

        trainCols, calendar, prices = FManager.load_objs(PRICE_CAL_TRAIN_DF_FP)
        model = load_model(NN_MODEL_FP)

        cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday',
                    'month', 'year', 'event_name_1', 'event_name_2', 'event_type_1',
                    'event_type_2', 'week', 'quarter', 'mday']
        cont_cols = ['sell_price', 'lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7',
                     'rmean_7_28', 'rmean_28_28']

        # PREDICTIONS
        self.lg.debug("making predictions...")

        for icount, (alpha, weight) in enumerate(zip(ALPHAS, WEIGHTS)):
            te = Shaper.create_ds(TR_LAST, MAX_LAGS, calendar, prices)
            cols = [f"F{i}" for i in range(1, 29)]

            for tdelta in range(0, 28):
                day = FDAY + timedelta(days=tdelta)
                self.lg.debug("%s, %s" % (tdelta, day))
                tst = te[(te['date'] >= day - timedelta(days=MAX_LAGS)) & (te['date'] <= day)].copy()
                Shaper.create_features(self.lg, tst)
                tst = tst.loc[tst['date'] == day, trainCols]

                k_tst =self._make_keras_input(tst, cat_cols)
                k_tst = self._make_keras_input(tst, cont_cols, k_tst)

                te.loc[te['date'] == day, "sales"] = alpha * model.predict(k_tst)  # magic multiplier by kyakovlev

            te_sub = te.loc[te['date'] >= FDAY, ["id", "sales"]].copy()
            te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
            te_sub = te_sub.set_index(["id", "F"]).unstack()["sales"][cols].reset_index()
            te_sub.fillna(0., inplace=True)
            te_sub.sort_values("id", inplace=True)
            te_sub.reset_index(drop=True, inplace=True)
            te_sub.to_csv(f"submissions/submission_{icount}.csv", index=False)

            if icount == 0:
                sub = te_sub
                sub[cols] *= weight
            else:
                sub[cols] += te_sub[cols] * weight

            self.lg.debug("iteration: %d, alpha: %f, weight: %f" %icount, alpha, weight)

        self.lg.debug("creating csv file submissions")
        sub2 = sub.copy()
        sub2["id"] = sub2["id"].str.replace("validation", "evaluation")
        sub = pd.concat([sub, sub2], axis=0, sort=False)
        sub.to_csv(SUBMISSION_FP, index=False)

    def _action_handle(self, actions, cuda):

        if "preprocess" in actions:
            self.preprocess()
        if "train" in actions:
            self.train(cuda)
        if "predict" in actions:
            self.predict()


    def run(self, actions=["train"], cuda=False):

        cpus = tf.config.experimental.list_physical_devices('CPU')

        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus and cuda:

            tf.config.set_soft_device_placement(True)
            tf.debugging.set_log_device_placement(True)

            gpus = tf.config.experimental.list_physical_devices('GPU')

            for gpu in gpus:
                self.lg.debug("running on the GPU: %s..." % (str(gpu)))
                try:
                    with tf.device(gpu.name.replace('physical_device:', '')):
                        self._action_handle(actions, cuda)
                except RuntimeError as e:
                    print(e)
                    traceback.print_tb(e)
                    traceback.print_stack(e)
        else:
            self.lg.debug("running on the cpu...")
            self._action_handle(actions, cuda)