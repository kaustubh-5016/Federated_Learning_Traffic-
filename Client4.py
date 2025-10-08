import os
import sys
from Model import *
from PythonUtils import *
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':
    filename = str(sys.argv[1])
    client4_space = space_path('client4_space')
    df = load_data(client4_space, 'STTLng')
    train_x, train_y, test_x, test_y = make_rnn_data('STTLng', df, .7, 70)
    val_x = test_x[:-100]
    val_y = test_y[:-100]
    test_x = test_x[-100:]
    test_y = test_y[-100:]
    model_path = os.path.join(client4_space, 'client4_model.h5')
    weight_path = os.path.join(client4_space, filename)
    if os.path.exists(model_path) and os.path.exists(weight_path):
        cm = load_model(model_path)
        cm.load_weights(weight_path)
        find_mean_squared_error(cm, '4', test_y, test_x)
        history = train_model_without_callback(cm, 1, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        cm.save_weights(os.path.join(client4_space, 'client4_weights.h5'))
        cm.save(model_path)
        update_and_save_the_learning_curve(client4_space, history, 'loss', 'STTLng')
    else:
        m = load_model(weight_path)
        history = train_model_without_callback(m, 1, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        m.save_weights(os.path.join(client4_space, 'client4_weights.h5'))
        m.save(model_path)
        update_and_save_the_learning_curve(client4_space, history, 'loss', 'STTLng')
    delete_file(weight_path)
