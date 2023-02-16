import numpy as np
import os
import glob

import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ttest_ind
from tifffile import imread

# ..........torch imports............
from torchvision import transforms

# .... Captum imports..................
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset


# Input transformations

def transform(img):

    return transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )(img)

# Data handling
def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transform(img)


def get_tensor_from_EuroSAT_filename(filename):
    np_img = imread(filename)
    np_out_img = np_img[:, :, 1:4]
    np_out_img = np_out_img[:, :, ::-1].astype('float64')
    np_out_img /= 4095
    np_out_img *= 255
    img = Image.fromarray(np_out_img.astype('uint8')).convert('RGB')
    # modes: 
    # multi - stich images
    # single - mirror images
    mode = 'multi'
    # change fill_dir here
    fill_dir = 'D:/adml/data/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/Pasture/'
    if mode == 'single':
        img = transforms.Pad(padding=100, padding_mode='reflect').forward(img)
    else:
        image_files = np.random.choice(os.listdir(fill_dir), 15)
        img_size = img.size
        new_image = Image.new(
            'RGB', (4*img_size[0], 4*img_size[1]), (250, 250, 250))
        new_image.paste(img, (0, 0))
        pos = (img_size[0], 0)
        for image_file in image_files:
            np_img = imread(fill_dir + image_file)
            np_out_img = np_img[:, :, 1:4]
            np_out_img = np_out_img[:, :, ::-1].astype('float64')
            np_out_img /= 4095
            np_out_img *= 255
            img = Image.fromarray(np_out_img.astype('uint8')).convert('RGB')
            new_image.paste(img, pos)
            pos = list(pos)
            pos[0] += img_size[0]
            if pos[0] > 3*img_size[0]:
                pos[0] = 0
                pos[1] += img_size[1]
            pos = tuple(pos)
        img = new_image
    plt.imshow(img)
    return transform(img)


def load_image_tensors(class_name, root_path='anthroprotect/tiles/s2/', transform=True, start=0):
    filenames = glob.glob(root_path + class_name + '*.tif')

    tensors = []
    for filename in filenames[start:start+11]:
        np_img = imread(filename)
        np_out_img = np_img[:, :, 0:3]
        np_out_img = np_out_img[:, :, ::-1].astype('float64')
        #np_out_img -= np_out_img.min()
        #np_out_img /= np_out_img.max()
        np_out_img /= 4095
        np_out_img *= 255
        img = Image.fromarray(np_out_img.astype('uint8')).convert('RGB')
        tensors.append(transform(img) if transform else img)

    return tensors

# Concept assembly

def assemble_concept(name, id, concepts_path="data/tcav/image/concepts/"):
    if concepts_path == "data/eurosat/concepts/":
        concept_path = os.path.join(concepts_path, name) + "/"
        dataset = CustomIterableDataset(
            get_tensor_from_EuroSAT_filename, concept_path)
        concept_iter = dataset_to_dataloader(dataset, batch_size=5)
    else:
        concept_path = os.path.join(concepts_path, name) + "/"
        dataset = CustomIterableDataset(
            get_tensor_from_filename, concept_path)
        concept_iter = dataset_to_dataloader(dataset, batch_size=5)

    return Concept(id=id, name=name, data_iter=concept_iter)


# Testing

def assemble_scores_multiInput(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    sub_score_list = []
    for concepts in experimental_sets:
        for score in scores:
            sub_score_list.append(
                score["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx])
        score_list.append(np.mean(sub_score_list))

    return score_list


# Plot test results

def plot_results(layers, experimental_sets, scores, n, metric='sign_count'):
    vals = []
    stds = []
    significant = []
    label = []
    esl_concept = experimental_sets[0: n]
    esl_random = experimental_sets[n:]
    for layer in layers:
        P1_concept = assemble_scores_multiInput(
            scores, esl_concept, 0, layer, metric)
        P2_random = assemble_scores_multiInput(
            scores, esl_random, 0, layer, metric)

        _, pval = ttest_ind(P1_concept, P2_random)

        label.append(layer)

        if pval > 0.05:
            vals.append(0.01)
            stds.append(0)
            significant.append(False)
        else:
            vals.append(np.mean(P1_concept))
            stds.append(np.std(P1_concept))
            significant.append(True)
    plt.figure()
    plt.bar(label, vals, yerr=stds)
    plt.title('TCAV Scores for each layer')
    plt.ylabel('TCAV Score')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
