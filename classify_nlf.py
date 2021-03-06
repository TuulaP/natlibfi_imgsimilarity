from __future__ import absolute_import, division, print_function

#import tensorflow as tf
import tensorflow.compat.v1 as tf
from six.moves import urllib
import numpy as np
from collections import defaultdict
import psutil
import json
import glob
import tarfile
import sys
import re
import os.path


####Delete all flags before declare#####
# https://stackoverflow.com/a/51211037/364931
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

"""

This is a modification of the classify_images.py
script in Tensorflow. The original script produces
string labels for input images (e.g. you input a picture
of a cat and the script returns the string "cat"); this
modification reads in a directory of images and
generates a vector representation of the image using
the penultimate layer of neural network weights.

Usage: python classify_images.py "../image_dir/*.jpg"

"""

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', './core',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")


# because of https://github.com/tensorflow/tensorflow/issues/17702#issuecomment-387335646
#tf.app.flags.DEFINE_string('f', '', 'kernel')


# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.

    # because of https://github.com/tensorflow/tensorflow/issues/17702#issuecomment-387335646
    #tf.app.flags.DEFINE_string('f', '', 'kernel')

    with tf.gfile.GFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_images(image_list, output_dir):
    """Runs inference on an image list.

    Args:
      image_list: a list of images.
      output_dir: the directory in which image vectors will be saved

    Returns:
      image_to_labels: a dictionary with image file keys and predicted
        text label values
    """
    image_to_labels = defaultdict(list)
    outfile_name = ""

    create_graph()

    # print("Starting run_inference...")

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        for image_index, image in enumerate(image_list):
            try:
                print("Parsing", image_index, image, "\n")
                if not tf.gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)

                with tf.gfile.GFile(image, 'rb') as f:
                    image_data = f.read()
                    # print("IMg read.")
                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})

                    predictions = np.squeeze(predictions)

                    ###
                    # Get penultimate layer weights
                    ###

                    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                    feature_set = sess.run(feature_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    feature_vector = np.squeeze(feature_set)
                    # print("f_veatures done. img: {0}".format(image))
                    outfile_name = os.path.basename(image) + ".npz"
                    # print("123", output_dir)
                    out_path = os.path.join(output_dir, outfile_name)
                    # print("paths figured: {0} to {1}".format(
                    #    outfile_name, out_path))
                    np.savetxt(out_path, feature_vector, delimiter=',')
                    # print("SAved.")
                    # Creates node ID --> English string lookup.
                    node_lookup = NodeLookup()
                    # print("Features figured... ")

                    top_k = predictions.argsort(
                    )[-FLAGS.num_top_predictions:][::-1]
                    for node_id in top_k:
                        human_string = node_lookup.id_to_string(node_id)
                        score = predictions[node_id]
                        # print("results for", image)
                        # print('%s:  %s (score = %.5f)' %
                        #      (image, human_string, score))
                        # print("\n")

                        image_to_labels[image].append(
                            {
                                "labels": human_string,
                                "score": str(score)
                            }
                        )

                # close the open file handlers
                # print("Closing file handlers... ")
                # proc = psutil.Process()
                # print("proc: ", proc)
                # open_files = proc.open_files()
                # print("Open files: ", openfiles)

                # for open_file in open_files:
                #     file_handler = getattr(open_file, "fd")
                #     print("Processing.. {0}, eli {1}. ".format(
                #         open_file, open_file['path']))

                # print("file handlers closed... ")
            except:
                print('could not process image index',
                      image_index, 'image', image)

    #print("run_inference done.")

    return (image_to_labels, outfile_name)


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    maybe_download_and_extract()
    if len(sys.argv) < 2:
        print("please provide a glob path to one or more images, e.g.")
        print("python classify_image_modified.py '../cats/*.jpg'")
        sys.exit()

    else:
        output_dir = "nlf_vectors"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images = glob.glob(sys.argv[1], recursive=True)
        image_to_labels = run_inference_on_images(images, output_dir)

        with open("image_to_labels.json", "w") as img_to_labels_out:
            json.dump(image_to_labels, img_to_labels_out)

        print("all done")


if __name__ == '__main__':

    del_all_flags(tf.flags.FLAGS)

    tf.app.run()
