import apache_beam as beam
import os
import datetime
import subprocess
import tensorflow as tf
import numpy as np


def _string_feature(value):
    return tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                                        value=[value.encode('utf-8')]
                                                         )
                           )


def _int64_feature(value):
    return tf.train.Feature(
                            int64_list=tf.train.Int64List(value=value)
                           )


def _float_feature(value):
    return tf.train.Feature(
                            float_list=tf.train.FloatList(value=value)
                           )


def read_and_decode(filename):
    IMG_CHANNELS = 3  # RGB; 4th channel is for opacity
    print(f'Reading {filename}') 
    img = tf.io.read_file(filename)  # read a sequence of bytes
    # convert to pixels (by using lookup tables)
    img = tf.image.decode_jpeg(
                               img,
                               channels=IMG_CHANNELS
                              )
    # scaling of pixel values from [0,255] to [0,1]
    img = tf.image.convert_image_dtype(img,
                                       tf.float32)  # conversion to float
    return img


def create_tfrecord(filename,
                    label,
                    label_int):
    print(filename)
    img = read_and_decode(filename)
    dims = img.shape
    img = tf.reshape(img, [-1])  # flatten to 1D array
    return tf.train.Example(
                            features=tf.train.Features(feature={
                                                                'image': _float_feature(img),
                                                                'shape': _int64_feature([dims[0],
                                                                                         dims[1],
                                                                                         dims[2]]),
                                                                'label': _string_feature(label),
                                                                'label_int': _int64_feature([label_int])
                                                               }
                                                       )
                            ).SerializeToString()


def assign_record_to_split(rec):  # 60:20:20 train:val:test split
    rnd = np.random.rand()
    if rnd <= 0.6:
        return ('train', rec)
    elif rnd <= 0.8:
        return ('valid', rec)
    return ('test', rec)


def yield_records_for_split(x, desired_split):  # whether train/val/test
    split, rec = x
    # print(split, desired_split, split == desired_split)
    if split == desired_split:
        yield rec


def write_records(OUTPUT_DIR,
                  splits,
                  split):
    nshards = 2
    # Since floats occupy more space, hence use compression/gzip
    _ = (splits
         | 'only_{}'.format(split) >> beam.FlatMap(
                                                   lambda x: yield_records_for_split(x, split)
                                                  )
         | 'write_{}'.format(split) >> beam.io.tfrecordio.WriteToTFRecord(
                                                                          os.path.join(OUTPUT_DIR, split),
                                                                          file_name_suffix='.gz',
                                                                          num_shards=nshards
                                                                         )
        )


if __name__ == '__main__':

    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    JOBNAME = (
               'jpg-to-tfrecord-' + timestamp  # only [-a-z0-9] allowed
              )
    OUTPUT_DIR = "gs://fire_detection_anurag/tfrecords"
    temp_dir = "gs://fire_detection_anurag/temp_dir"
    RUNNER = 'DataflowRunner'
    LABELS = ["Fire", "No_Fire"]  # <-- Change this
    PROJECT = "kubeflow-1-0-2"  # <-- Change this
    region = 'us-central1'
    file_csv = "gs://fire_detection_anurag/fire_dataset_AutoML.csv"  # <-- Change this
    dependencies = "requirements.txt"
    max_num_workers = 3

    # clean-up output directory since Beam will name files like 0000-of-0004 etc.
    # and this could cause confusion if earlier run has 0000-of-0005, for example
    try:
        subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
        subprocess.check_call('gsutil -m rm -r {}'.format(temp_dir).split())
    except subprocess.CalledProcessError:
        pass

    options = {
                'staging_location': os.path.join(temp_dir, 'staging'),
                'temp_location': os.path.join(temp_dir, 'tmp'),
                'job_name': JOBNAME,
                'project': PROJECT,
                'max_num_workers': max_num_workers,  # autoscaling
                'region': region,
                'teardown_policy': 'TEARDOWN_ALWAYS',
                'save_main_session': True
            }

    opts = beam.pipeline.PipelineOptions(flags=[], **options)

    with beam.Pipeline(RUNNER, options=opts) as p:
        splits = (p
                  | 'read_csv' >> beam.io.ReadFromText(file_csv)
                  | 'parse_csv' >> beam.Map(lambda line: line.split(','))
                  | 'convert_to_tfr' >> beam.Map(lambda x: create_tfrecord(
                                                                        x[0],
                                                                        x[1],
                                                                        LABELS.index(x[1])))
                  | 'train_val_test split' >> beam.Map(assign_record_to_split)
                  )

        for split in ['train', 'valid', 'test']:
            write_records(OUTPUT_DIR,
                          splits,
                          split)