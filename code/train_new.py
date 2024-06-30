import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
import segmentation_models as sm
from train_utils import CustomDataGen, MyModelCheckpoint
from train_utils import create_if_not_exist

import json


## hyperparameters
hyperparameters = dict(
    epochs=200,
    image_size_input=64, 
    n_c_out=3,
    n_c_in=4,
    batch_size=32, 
    lr=0.0001, 
    l2reg=.1, 
    activation='linear', 
    weight='mask',
    save_from_epoch=90,
    model_name = 'UNet',
    backbone_name = 'resnet18' 
)

model_name = hyperparameters['model_name']
model_f = eval('sm.' + model_name)
backbone_name = hyperparameters['backbone_name']


## model and data parameters
image_size_input = hyperparameters['image_size_input']
n_c_in = hyperparameters['n_c_in']
n_c_out = hyperparameters['n_c_out']
lr = hyperparameters['lr']


image_shape = (image_size_input,image_size_input, n_c_in)
label_shape = (image_size_input,image_size_input, n_c_out)

epoch = hyperparameters['epochs']
batch_size = hyperparameters['batch_size']

c_in = image_shape[-1]
num_class = label_shape[-1]
l2reg = hyperparameters['l2reg']

activation = hyperparameters['activation']
## generators
PARENT_FOLDER = os.getcwd()
paths_list_x = glob.glob(os.path.join(PARENT_FOLDER, 'data', 'train', 'x', '*.tif'))
paths_list_y = glob.glob(os.path.join(PARENT_FOLDER, 'data', 'train', 'y', '*.tif'))
paths_list_w = glob.glob(os.path.join(PARENT_FOLDER, 'data', 'train', 'w', '*.tif'))

paths_list_x_train, paths_list_x_val, paths_list_y_train, paths_list_y_val, paths_list_w_train, paths_list_w_val, = train_test_split(paths_list_x, paths_list_y, paths_list_w, test_size=0.2)

x_train_gen = CustomDataGen(paths_list_x_train, batch_size=batch_size, epoch=epoch, x=True)
y_train_gen = CustomDataGen(paths_list_y_train, batch_size=batch_size, epoch=epoch)
w_train_gen = CustomDataGen(paths_list_w_train, batch_size=batch_size, epoch=epoch, w=True)

x_val_gen =  CustomDataGen(paths_list_x_val, batch_size=batch_size, epoch=epoch, x=True)
y_val_gen =  CustomDataGen(paths_list_y_val, batch_size=batch_size, epoch=epoch)
w_val_gen =  CustomDataGen(paths_list_w_val, batch_size=batch_size, epoch=epoch, w=True)

assert len(x_train_gen.channels_to_select) == c_in

# x = x_train_gen[200]
# print('x_shape: ', x.shape)
# y = y_train_gen[200]
# print('y_shape: ', y.shape)
# w = w_train_gen[200]
# print('w_shape: ', w.shape)

print('train', x_train_gen.n)
print('val', x_val_gen.n)

# import sys
# sys.exit()



if __name__ == '__main__': #for model_f, model_name, backbone_name in zip(models, model_names, backbone_names):

    model = model_f(backbone_name=backbone_name, encoder_weights=None, input_shape=(None, None, c_in), classes=num_class, activation=activation)
    
    if l2reg != 0:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(l2reg)

    #print(model.summary())
    #tf.keras.utils.plot_model(model, to_file='plot.png', show_shapes=True)
    # import sys
    # sys.exit()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
        loss=tf.keras.losses.CosineSimilarity(), #custom_cosine_loss, 
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse"), ],
    )

    # print(model.summary())
    # import sys
    # sys.exit()


    create_if_not_exist(os.path.join(PARENT_FOLDER, 'results'))
    # check points 
    
    with open(os.path.join(PARENT_FOLDER, 'results', 'last_model_number.txt')) as f:  
        last_model_num = f.read()

    model_save_path = os.path.join(PARENT_FOLDER, 'results', str(int(last_model_num)+1)+ '_' + model_name + '_' + backbone_name) 

    with open(os.path.join(PARENT_FOLDER, 'results', 'last_model_number.txt'), 'w') as f:
        f.write(str(int(last_model_num)+1))
        
        
    hyperparameters_path = os.path.join(model_save_path, 'hyperparameters.json')

    hyperparameters['model_name'] = model_name
    hyperparameters['backbone_name'] = backbone_name
    
    create_if_not_exist(model_save_path)
    create_if_not_exist(os.path.join(model_save_path, 'model'))
    
    with open(hyperparameters_path, 'w') as file:
        file.write(json.dumps(hyperparameters))
  
    ckp_path = os.path.join(model_save_path, 'model', 'weights-improvement-{epoch:02d}.hdf5')
        
    filename = os.path.join(model_save_path, 'train.csv')

    history_logger = CSVLogger(filename, separator=",", append=True)

    checkpoint = MyModelCheckpoint(hyperparameters['save_from_epoch'], ckp_path)
    # earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    callbacks_list = [checkpoint, history_logger, ]


        
    # train the model       
    # t1 = time.time()

    hist = model.fit(
        x=zip(x_train_gen, y_train_gen, w_train_gen),
        batch_size=batch_size,
        epochs=epoch,
        
        verbose=0,
        shuffle=True,
        validation_data=zip(x_val_gen, y_val_gen, w_val_gen),
        steps_per_epoch = x_train_gen.n // batch_size,
        validation_steps = x_val_gen.n // batch_size,
        callbacks=callbacks_list,
    )
    
    # print('-------------------------------------------------------------------------------------------------------------------------------------------------')
    # print('-------------------------------------------------------------------------------------------------------------------------------------------------')
    # print('---------------------------------------------------------------- GOING TO -----------------------------------------------------------------------')
    # print('--------------------------------------------------------------- NEXT MODEL ----------------------------------------------------------------------')
    # print('-------------------------------------------------------------------------------------------------------------------------------------------------')
    # print('-------------------------------------------------------------------------------------------------------------------------------------------------')
