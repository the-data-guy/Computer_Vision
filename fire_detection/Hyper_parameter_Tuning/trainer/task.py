import argparse
import hypertune
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.data.experimental import AUTOTUNE
import datetime as dt

IMG_HEIGHT = 448  # TODO
IMG_WIDTH = 448
IMG_CHANNELS = 3
CLASS_NAMES = 'Fire No-Fire'.split()

tfrecords_in_gcs = "gs://fire_detection_anurag/tfrecords/"
PATTERN_SUFFIX = '-*'

training_data_tfr = tfrecords_in_gcs  + "train" + PATTERN_SUFFIX
validation_data_tfr = tfrecords_in_gcs + "valid" + PATTERN_SUFFIX

NUM_EPOCHS = 9


def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--learning_rate',
                        required=True,
                        type=float,
                        help='learning rate'
                       )
    parser.add_argument(
                        '--momentum',
                        required=True,
                        type=float,
                        help='SGD momentum value'
                       )
    parser.add_argument(
                        '--num_hidden',
                        required=True,
                        type=int,
                        help='no. of nodes in last layer'
                       )
#     parser.add_argument(
#                         '--batch_size',
#                         required=True,
#                         type=int,
#                         help='for training in batches'
#                        )
    
    args = parser.parse_args()
    return args


class _Preprocessor:    
    def __init__(self):
        # nothing to initialize
        pass
    
    
    def read_from_tfr(self, proto):
        feature_description = {
                                'image': tf.io.VarLenFeature(tf.float32),
                                'shape': tf.io.VarLenFeature(tf.int64),
                                'label': tf.io.FixedLenFeature([], tf.string,
                                                               default_value=''),
                                'label_int': tf.io.FixedLenFeature([], tf.int64,
                                                                   default_value=0),
                              }
        rec = tf.io.parse_single_example(
                                        proto,
                                        feature_description
                                        )
        shape = tf.sparse.to_dense(rec['shape'])
        img = tf.reshape(tf.sparse.to_dense(rec['image']), shape)
        label_int = rec['label_int']
        return img, label_int
    
    
    def preprocess(self, img):
        return tf.image.resize_with_pad(img,
                                        IMG_HEIGHT,
                                        IMG_WIDTH)
    
    
# split the files into two halves and interleaves datasets
def create_preproc_dataset(pattern):
    """
    Does interleaving, parallel calls, prefetch, batching
    Caching is not a good idea on large datasets.
    """
    preproc = _Preprocessor()
    files = [filename for filename in tf.random.shuffle(tf.io.gfile.glob(pattern))]
    
    if len(files) > 1:
        print("Interleaving the reading of {} files.".format(len(files)))
        
        
        def _create_half_ds(x):
            if x == 0:
                half = files[:(len(files)//2)]
            else:
                half = files[(len(files)//2):]
                
            return tf.data.TFRecordDataset(half,
                                           compression_type='GZIP')
        
        
        ds = tf.data.Dataset.range(2).interleave(_create_half_ds,
                                                 num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.TFRecordDataset(files,
                                     compression_type='GZIP')
        
        
    def _preproc_img_label(img, label):
        return (preproc.preprocess(img), label)
    
    
    ds = (ds
           .map(preproc.read_from_tfr,
                num_parallel_calls=AUTOTUNE)
           .map(_preproc_img_label,
                num_parallel_calls=AUTOTUNE)
           .shuffle(200)  # TODO
           .prefetch(AUTOTUNE)
         )
    
    return ds


def create_model(l1, l2,
                 num_hidden):
    
    regularizer = tf.keras.regularizers.l1_l2(l1, l2)
        
    layers = [
              tf.keras.layers.experimental.preprocessing.RandomCrop(
                                                                  height=IMG_HEIGHT//2,
                                                                  width=IMG_WIDTH//2,
                                                                  input_shape=(IMG_HEIGHT,
                                                                               IMG_WIDTH,
                                                                               IMG_CHANNELS),
                                                                  name='random/center_crop'
                                                                   ),
              tf.keras.layers.experimental.preprocessing.RandomFlip(
                                                                  mode='horizontal',
                                                                  name='random_lr_flip/none'
                                                                   ),
              hub.KerasLayer(
                              "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                              trainable=False,
                              name='mobilenet_embedding'
                            ),
              tf.keras.layers.Dense(num_hidden,
                                    kernel_regularizer=regularizer, 
                                    activation=tf.keras.activations.relu,
                                    name='dense_hidden'),
              tf.keras.layers.Dense(len(CLASS_NAMES),  # len(CLASS_NAMES) vs. 1
                                    kernel_regularizer=regularizer,
                                    activation='softmax',  # softmax vs. sigmoid
                                    name='fire_detector')
            ]

    # create model
    return tf.keras.Sequential(layers,
                               name='fire_detection')


def train_and_evaluate(
#                        strategy,
#                        batch_size,
                       lrate,
                       momentum,
                       num_hidden,
                       l1 = 0.,
                       l2 = 0.,
                      ):
    
    # callbacks
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                                                         monitor='val_accuracy',
                                                         mode='max',
                                                         patience=3
                                                        )
    
    # model training
#     with strategy.scope():
#         model = create_model(l1, l2, num_hidden)
    model = create_model(l1, l2, num_hidden)
    
    model.compile(
                  optimizer=tf.keras.optimizers.SGD(
                                                    learning_rate=lrate,
                                                    momentum=momentum    
                                                   ),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                                                     from_logits=False
                                                                    ),
                  metrics=['accuracy']  # Our dataset/classes are not really imbalanced.
                 )
    
    return model


def main():
    args = get_args()
    
#     strategy = tf.distribute.MirroredStrategy()

    # Scale batch size
    batch_size = 32
#     GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
#     GLOBAL_BATCH_SIZE = args.batch_size

    train_dataset = create_preproc_dataset(
                                           training_data_tfr
                                          ).batch(batch_size)
    eval_dataset = create_preproc_dataset(
                                          validation_data_tfr
                                         ).batch(batch_size)
    
    # build model
    model = train_and_evaluate(
#                                strategy,
#                                batch_size = GLOBAL_BATCH_SIZE,
                               lrate = args.learning_rate,
                               momentum = args.momentum,
                               num_hidden = args.num_hidden,
                               l1 = 0.,
                               l2 = 0.,
                              )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                                                         monitor='val_accuracy',
                                                         mode='max',
                                                         patience=3
                                                        )
    
    # train model
    history = model.fit(
                        train_dataset, 
                        validation_data=eval_dataset,
                        epochs=NUM_EPOCHS,
                        callbacks=[
                                   early_stopping_cb,
                                  ]
                       )
    
    # DEFINE METRIC
    hp_metric = history.history['val_accuracy'][-1]  # metric to be optimized

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
                                            hyperparameter_metric_tag='accuracy',
                                            metric_value=hp_metric,
                                            global_step=NUM_EPOCHS
                                            )

if __name__ == "__main__":
    main()