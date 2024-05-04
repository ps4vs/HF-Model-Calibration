import numpy
import torch
import metrics
import visualization
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


def prepare_model(ckpt_name):
    # load_dataset, load_model
    image_processor = AutoImageProcessor.from_pretrained(ckpt_name)
    model = AutoModelForImageClassification.from_pretrained(ckpt_name)
    return image_processor, model

def prepare_dataset(dataset_loc, split, num_images):
    dataset = load_dataset(dataset_loc, split=split, streaming=True, trust_remote_code=True).with_format('torch')
    count = min(50000, num_images*2)
    newdataset = dataset.shuffle(seed=42, buffer_size=count).take(count)
    return newdataset

def evaluate(image_processor, model, dataset, num_images, save, save_dir='.'):
    # evaluate, generate_plots.
    correct = 0
    total = 0

    logits_list = []
    labels_list = []
    broken_ids = []

    with torch.no_grad():

        for i, data in enumerate(tqdm(iter(dataset))):
            try:
                inputs = image_processor(data['image'], return_tensors="pt")
                if inputs['pixel_values'].shape == 4:
                    inputs['pixel_values'] = inputs['pixel_values'].squeeze(1)
            except:
                broken_ids.append(i)
                continue
            outputs = model(**inputs)
            logits = outputs.logits
            labels = data['label'].unsqueeze(dim=0)

            logits_list.append(logits)
            labels_list.append(labels)

            output_probs = F.softmax(logits,dim=1)
            probs, predicted = torch.max(output_probs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if len(labels_list)==num_images:
                break
        print('Broken images, which are not included: %d' % (len(broken_ids)))
        print(f'Accuracy on the {len(labels_list)} validation images: {(100 * correct / total)}')
        print(f'number of images used {total}') 

        logits_np = torch.cat(logits_list).numpy()
        labels_np = torch.cat(labels_list).numpy()
        if save:
            numpy.save(save_dir + '/logits_np.npy', logits_np)
            numpy.save(save_dir + '/labels_np.npy', labels_np)
    return logits_np, labels_np

def calibrate_and_plot(logits_np, labels_np, save, save_dir='.'):
    ece_criterion = metrics.ECELoss()
    sce_criterion = metrics.SCELoss()

    ece = (ece_criterion.loss(logits_np,labels_np, 15))
    sce = (sce_criterion.loss(logits_np,labels_np, 15))
    print('ECE: %f' % ece)
    print('SCE: %f' % sce)
    if save:
        numpy.save(save_dir + '/ece.npy', ece)
        numpy.save(save_dir + '/sce.npy', sce)
    conf_hist = visualization.ConfidenceHistogram()
    plt_test = conf_hist.plot(logits_np,labels_np,title="Confidence Histogram")
    if save:
        plt_test.savefig(save_dir + '/conf_histogram_test.png',bbox_inches='tight')
    plt_test.show()

    rel_diagram = visualization.ReliabilityDiagram()
    plt_test_2 = rel_diagram.plot(logits_np,labels_np,title="Reliability Diagram")
    if save:
        plt_test_2.savefig(save_dir + '/rel_diagram_test.png',bbox_inches='tight')
    plt_test_2.show()