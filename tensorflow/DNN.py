import tensorflow as tf


def DNN(dimensions, model_name = 'DNN'):
        
    tf.keras.backend.clear_session() # We don't want to mess up with model's name
    
    '''
    model = tf.keras.Sequential([
        tf.keras.Input(shape = (10,)),
        
        tf.keras.layers.Dense(1000),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.Dense(1000),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        tf.keras.layers.Dense(1),
    ], name = model_name)
    '''
    
    inputs = tf.keras.Input(shape = (dimensions[0]))
    outputs = inputs
    for dimension in dimensions[1:-1]:
        outputs = tf.keras.layers.Dense(dimension)(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.ReLU()(outputs)
    outputs = tf.keras.layers.Dense(dimensions[-1])(outputs)
    
    model = tf.keras.Model(inputs, outputs, name = model_name)
    
    return model


if __name__ == '__main__':
    
    model = DNN(dimensions = [10, 1000, 1000, 1])
    model.summary()
    '''
    Model: "DNN"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 10)]              0         
                                                                     
     dense (Dense)               (None, 1000)              11000     
                                                                     
     batch_normalization (BatchN  (None, 1000)             4000      
     ormalization)                                                   
                                                                     
     re_lu (ReLU)                (None, 1000)              0         
                                                                     
     dense_1 (Dense)             (None, 1000)              1001000   
                                                                     
     batch_normalization_1 (Batc  (None, 1000)             4000      
     hNormalization)                                                 
                                                                     
     re_lu_1 (ReLU)              (None, 1000)              0         
                                                                     
     dense_2 (Dense)             (None, 1)                 1001      
                                                                     
    =================================================================
    Total params: 1,021,001
    Trainable params: 1,017,001
    Non-trainable params: 4,000
    _________________________________________________________________
    '''