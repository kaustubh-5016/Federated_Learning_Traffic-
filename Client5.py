import os
import sys
from Model import *
from PythonUtils import *
from tensorflow.python.keras.models import load_model

if __name__ == '__main__':
    filename = str(sys.argv[1])
    c_path = os.getcwd()
    client5_space = c_path + '\\data\\client5_space\\'
    df = load_data(client5_space,  'WASHng')
    train_x, train_y, test_x, test_y = make_rnn_data('WASHng',df,  .7,70)
    val_x = test_x[:-100]
    val_y = test_y[:-100]
    test_x = test_x[-100:]
    test_y = test_y[-100:]
    print(train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)
    if os.path.exists(client5_space + 'client5_model.h5') and os.path.exists(client5_space + filename):
        cm = load_model(client5_space + 'client5_model.h5')
        cm.load_weights(client5_space + filename)
        history = train_model_without_callback(cm, 1, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        cm.save_weights(client5_space + 'client5_weights.h5')
        cm.save(client5_space + 'client5_model.h5')
        update_and_save_the_learning_curve(client5_space, history, 'loss','WASHng')
        find_mean_squared_error(cm, '5', test_y, test_x)
    else:
        m = load_model(client5_space + filename)
        history = train_model_without_callback(m, 1, 256, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)
        m.save_weights(client5_space + 'client5_weights.h5')
        m.save(client5_space + 'client5_model.h5')
        update_and_save_the_learning_curve(client5_space, history, 'loss','WASHng')
    delete_file(client5_space + filename)
