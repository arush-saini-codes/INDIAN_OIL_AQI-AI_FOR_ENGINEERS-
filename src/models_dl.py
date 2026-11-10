"""Neural network architectures for Phase 6 Deep Learning.
Implements Stacked LSTM, Stacked GRU, Bidirectional LSTM, and CNN-LSTM.
Supports single-step (1-day) and direct multi-step (7-day) outputs,
regularization, and Monte Carlo dropout for predictive uncertainty.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


def build_stacked_lstm(
    input_shape,
    units_l1=64,
    units_l2=64,
    dropout=0.2,
    recurrent_dropout=0.0,
    l2_reg=0.0,
    output_dim=1,
    lr=1e-3,
    loss_fn='mse'
):
    """Stacked 2-layer LSTM model."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(
        units_l1,
        return_sequences=True,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=reg,
        name='lstm_layer1'
    )(inp)
    if dropout > 0:
        x = layers.Dropout(dropout, name='dropout_1')(x)
        
    x = layers.LSTM(
        units_l2,
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=reg,
        name='lstm_layer2'
    )(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name='dropout_2')(x)
        
    out = layers.Dense(output_dim, kernel_regularizer=reg, name='output_dense')(x)
    
    model = Model(inputs=inp, outputs=out, name='Stacked_LSTM')
    loss = tf.keras.losses.Huber(delta=1.0) if loss_fn.lower() == 'huber' else 'mse'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=['mae'])
    return model


def build_stacked_gru(
    input_shape,
    units_l1=64,
    units_l2=64,
    dropout=0.2,
    recurrent_dropout=0.0,
    l2_reg=0.0,
    output_dim=1,
    lr=1e-3,
    loss_fn='mse'
):
    """Stacked 2-layer GRU model."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(
        units_l1,
        return_sequences=True,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=reg,
        name='gru_layer1'
    )(inp)
    if dropout > 0:
        x = layers.Dropout(dropout, name='dropout_1')(x)
        
    x = layers.GRU(
        units_l2,
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=reg,
        name='gru_layer2'
    )(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name='dropout_2')(x)
        
    out = layers.Dense(output_dim, kernel_regularizer=reg, name='output_dense')(x)
    
    model = Model(inputs=inp, outputs=out, name='Stacked_GRU')
    loss = tf.keras.losses.Huber(delta=1.0) if loss_fn.lower() == 'huber' else 'mse'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=['mae'])
    return model


def build_bilstm(
    input_shape,
    units_l1=32,
    units_l2=32,
    dropout=0.2,
    recurrent_dropout=0.0,
    l2_reg=0.0,
    output_dim=1,
    lr=1e-3,
    loss_fn='mse'
):
    """Bidirectional 2-layer LSTM model (units per direction match parameter count)."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    inp = layers.Input(shape=input_shape)
    lstm_f1 = layers.LSTM(units_l1, return_sequences=True, recurrent_dropout=recurrent_dropout, kernel_regularizer=reg)
    x = layers.Bidirectional(lstm_f1, name='bilstm_layer1')(inp)
    if dropout > 0:
        x = layers.Dropout(dropout, name='dropout_1')(x)
        
    lstm_f2 = layers.LSTM(units_l2, return_sequences=False, recurrent_dropout=recurrent_dropout, kernel_regularizer=reg)
    x = layers.Bidirectional(lstm_f2, name='bilstm_layer2')(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name='dropout_2')(x)
        
    out = layers.Dense(output_dim, kernel_regularizer=reg, name='output_dense')(x)
    
    model = Model(inputs=inp, outputs=out, name='BiLSTM')
    loss = tf.keras.losses.Huber(delta=1.0) if loss_fn.lower() == 'huber' else 'mse'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=['mae'])
    return model


def build_cnn_lstm(
    input_shape,
    conv_filters=32,
    kernel_size=3,
    lstm_units=64,
    dropout=0.2,
    l2_reg=0.0,
    output_dim=1,
    lr=1e-3,
    loss_fn='mse'
):
    """Hybrid Conv1D + LSTM model."""
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    lookback = input_shape[0]
    
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu',
        kernel_regularizer=reg,
        name='conv1d'
    )(inp)
    
    if lookback >= 14:
        x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
        
    if dropout > 0:
        x = layers.Dropout(dropout, name='conv_dropout')(x)
        
    x = layers.LSTM(
        lstm_units,
        return_sequences=False,
        kernel_regularizer=reg,
        name='lstm_core'
    )(x)
    
    if dropout > 0:
        x = layers.Dropout(dropout, name='lstm_dropout')(x)
        
    out = layers.Dense(output_dim, kernel_regularizer=reg, name='output_dense')(x)
    
    model = Model(inputs=inp, outputs=out, name='CNN_LSTM')
    loss = tf.keras.losses.Huber(delta=1.0) if loss_fn.lower() == 'huber' else 'mse'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=['mae'])
    return model


def build_model(
    arch_type,
    input_shape,
    output_dim=1,
    units_l1=64,
    units_l2=64,
    dropout=0.2,
    recurrent_dropout=0.0,
    l2_reg=0.0,
    lr=1e-3,
    loss_fn='mse'
):
    """Unified dispatcher for building any architecture family."""
    arch = arch_type.lower()
    if 'gru' in arch:
        return build_stacked_gru(
            input_shape, units_l1, units_l2, dropout, recurrent_dropout, l2_reg, output_dim, lr, loss_fn
        )
    elif 'bilstm' in arch:
        # Scale units so total per layer matches units_l1
        u1 = max(16, units_l1 // 2)
        u2 = max(16, units_l2 // 2)
        return build_bilstm(
            input_shape, u1, u2, dropout, recurrent_dropout, l2_reg, output_dim, lr, loss_fn
        )
    elif 'cnn' in arch:
        return build_cnn_lstm(
            input_shape, conv_filters=32, kernel_size=3, lstm_units=units_l1,
            dropout=dropout, l2_reg=l2_reg, output_dim=output_dim, lr=lr, loss_fn=loss_fn
        )
    else:
        # Default Stacked LSTM
        return build_stacked_lstm(
            input_shape, units_l1, units_l2, dropout, recurrent_dropout, l2_reg, output_dim, lr, loss_fn
        )

