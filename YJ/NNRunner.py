from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers import Embedding, Add, Input, concatenate, SpatialDropout1D
from keras.layers import Flatten, Dropout, Dense, BatchNormalization, Concatenate
from tqdm.keras import TqdmCallback
import tensorflow as tf
import traceback
from YJ import Timer, FManager, Shaper
from YJ.environment import NN_MODEL_FP

def _make_embedded_layer(shape, name, max_val, embed_n):
    input = Input(shape=shape, name=name)
    embed = Embedding(max_val, embed_n)(input)

    return input, embed

def _make_dense_input_layer(shape, name, n_d_layers, act_type):
    input = Input(shape=shape, name=name)
    d_l = Dense(n_d_layers, activation=act_type)(input)
    d_l = BatchNormalization()(d_l)

    return input, d_l

def _make_keras_input(df, cols, X = None):
    if X is None:
        X = {}
        for each_col in cols:
            X[each_col] = df[each_col]
    else:
        for each_col in cols:
            X[each_col] = df[each_col]
    return X

def nn_train(lg):
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
        ds = Shaper.bottom_out_col(ds, cat_col)
        max_cat = Shaper.get_max(ds, cat_col)
        in_layer, embed_layer = _make_embedded_layer([1], cat_col, max_cat, n_embed_out)
        input_layers.append(in_layer)
        hid_layers.append(embed_layer)

    fe = concatenate(hid_layers)
    s_dout = SpatialDropout1D(0.1)(fe)
    x = Flatten()(s_dout)

    con_layers = []

    for con_col in cont_cols:
        in_layer, embed_layer = _make_dense_input_layer([1], con_col, n_embed_out, 'relu')
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

    X = _make_keras_input(ds, cat_cols)
    X = _make_keras_input(ds, cont_cols, X)

    checkpoint_name = 'models/weights-{epoch:03d}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    model.fit(X, y, batch_size=batch_size, use_multiprocessing=True,
              validation_split=.1, epochs=epochs, shuffle=True, verbose=0,
              callbacks=[TqdmCallback(verbose=2) ,checkpoint, es])

    model.save(model_fp)

def nn_make_eval_predict(lg):

    ds = FManager.load("objs/X_train.pkl")
    y = FManager.load("objs/y_train.pkl")

    cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday',
                'month', 'year', 'event_name_1', 'event_name_2', 'event_type_1',
                'event_type_2', 'week', 'quarter', 'mday']
    cont_cols = ['sell_price', 'lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7',
                 'rmean_7_28', 'rmean_28_28']

    batch_size = 1000

    for cat_col in cat_cols:
        ds = Shaper.bottom_out_col(ds, cat_col)

    X = _make_keras_input(ds, cat_cols)
    X = _make_keras_input(ds, cont_cols, X)

    model = tf.keras.models.load_model(NN_MODEL_FP)
    model.summary()

    # loss = model.evaluate(x=X, y=y, batch_size=batch_size, use_multiprocessing=True)
    # lg.debug("loss is: %f" % loss)
    lg.debug("making predictions...")
    predictions = model.predict(x=X, batch_size=batch_size, verbose=1,  use_multiprocessing=True)

    for i, prediction in enumerate(predictions):
        lg.debug("iteration: %d, prediction: %s" % (i, prediction))

def nn_run(lg):
    device_type = "GPU"

    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')

    if gpus and device_type == "GPU":
        for gpu in gpus:
            try:
                with tf.device(gpu.name.replace('physical_device:', '')):
                    nn_train(lg)
            except RuntimeError as e:
                print(e)
                traceback.print_tb(e)
                traceback.print_stack(e)
    else:
        nn_train(lg)