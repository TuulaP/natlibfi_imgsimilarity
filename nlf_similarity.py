# -*- coding: utf-8 -*-

import shutil
import numpy as np
import codecs
import glob
import json
import random
from nltk import ngrams
from scipy import spatial
from annoy import AnnoyIndex
from classify_nlf import run_inference_on_images
from pathlib import Path
import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# config
dims = 2048
n_nearest_neighbors = 5
trees = 10000
NPZDIR = "./tmp/"


# initial picture vectors to index

def initializepicvectors():

    # location where the feature descriptions of images exist.
    infiles = glob.glob(NPZDIR + '*.npz')

    for file_index, i in enumerate(infiles):
        file_vector = np.loadtxt(i)
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector
        t.add_item(file_index, file_vector)
    t.build(trees)

    return [t, file_index_to_file_name, file_index_to_file_vector]


def additemtoindex(item, t, file_index_to_file_name, file_index_to_file_vector):

    # see: https://github.com/spotify/annoy/issues/174#issuecomment-252616632
    #print("Unbuilding index to add one.")
    t.unbuild()
    #print("adding new one {0}".format(item))

    max = len(file_index_to_file_name)
    file_vector = np.loadtxt(item)
    t.add_item(max, file_vector)

    item = os.path.basename(item).split('.')[0]
    file_index_to_file_name[max] = item
    file_index_to_file_vector[max] = file_vector

    # print("Rebuilding...")
    t.build(trees)

    return [t, file_index_to_file_name, file_index_to_file_vector]


# takes the targetimage 'miimage' and compares it to the already 'known' images in order
# to find neighbors, which mean pictures that seem similar

def findsimilars(miimage, file_index_to_file_name, file_index_to_file_vector, t):
    images = []
    images.append(miimage)

    output_dir = "./tmp"

    (image_to_labels, npzfile) = run_inference_on_images(
        images, output_dir)  # this creates npz

    # with open("image_to_labels.json", "w") as img_to_labels_out:
    #            print("Dumping {0}".format(image_to_labels))
    #            json.dump(image_to_labels, img_to_labels_out)

    DIR = "./data/"
    target = DIR + miimage + ".npz"

    targetbase = target.replace(".jpg.npz", "").replace(DIR, "")
    print("Targetbase:{0}".format(targetbase))

    (t, file_index_to_file_name, file_index_to_file_vector) = additemtoindex(
        NPZDIR + npzfile, t, file_index_to_file_name, file_index_to_file_vector)

    # search _fk03927_1909-03_3_pg115_498017_pic01 , [0.18322894 0.62656009 0.30706945 ... 0.02660679 0.00194689 0.1605511 ]
    # neighbours _fk03927_1909-03_3_pg115_498017_pic01 , [0.18322894 0.62656009 0.30706945 ... 0.02660679 0.00194689 0.1605511 ]

    named_nearest_neighbors = []

    allfilenames = file_index_to_file_name.keys()

    for i in allfilenames:

        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]

        if (master_file_name in targetbase):  # no point to verify itself
            continue
        # else:
            #print("Calculating neighbourhood relations :)")
            # sys.exit(1)

        named_nearest_neighbors = []
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

        for j in nearest_neighbors:

            neighbor_file_name = file_index_to_file_name[j]
            neighbor_file_vector = file_index_to_file_vector[j]

            similarity = 1 - \
                spatial.distance.cosine(master_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0

            # 0.7599:  # 0.8, how much 'similarity' there should be.
            if rounded_similarity > 0.7500:
                named_nearest_neighbors.append({
                    'filename': neighbor_file_name,
                    'similarity': rounded_similarity
                })

        # for neigh in named_nearest_neighbors:
        #    #print("!", neigh)
        #    simis = len(named_nearest_neighbors)
        #    print("Closest similar: {2} --- {0}>>> {1} kpl; near.neigh: ".format(
        #        named_nearest_neighbors, simis, master_file_name))

        with open('nearest_neighbors/' + master_file_name + '.json', 'w') as out:
            json.dump(named_nearest_neighbors, out)
            #print("{0} written.".format(master_file_name+'.json'))

    #shutil.move(npzfile, "nlf_vectors/")
    #print("{0} moved with other vectors.".format(npzfile))

    return [named_nearest_neighbors, t, file_index_to_file_name, file_index_to_file_vector]


if __name__ == '__main__':

    import sys
    targetimg = sys.argv[1]

    t = AnnoyIndex(dims)

    # data structures
    file_index_to_file_name = {}
    file_index_to_file_vector = {}
    chart_image_positions = {}

    initializepicvectors()

    # the initial setup
    exampleset = glob.glob('data/*.jpg')
    #exampleset = []
    exampleset.append(targetimg)

    for targetimg in exampleset:  # ('hlo.jpg', 'orna2.jpg'):

        print("Items in file_index {0}".format(len(file_index_to_file_name)))

        (naapur, t, file_index_to_file_name, file_index_to_file_vector) = findsimilars(targetimg, file_index_to_file_name,
                                                                                       file_index_to_file_vector, t)
        print("For targetimg: {0}, found {1} neighbors".format(
            targetimg, len(naapur)))
        print("Neighbours are", naapur)

    print("All done.")
