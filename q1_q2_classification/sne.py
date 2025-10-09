'''
generate sne viz
'''
'''We can also visualize how the feature representations specialize for different classes. Take 1000
random images from the test set of PASCAL, and extract ImageNet (finetuned) features from
those images. Compute a 2D t-SNE (use sklearn) projection of the features, and plot them with
each feature color-coded by the GT class of the corresponding image. If multiple objects are
active in that image, compute the color as the “mean” color of the different classes active in that
image. Add a legend explaining the mapping from color to object class.'''

'''
1. get 1000 random images from test set of pascal
2. load model, change last fc to 
3. pass through model in batches
4. get down to dim50 with pca
5. get down to 2d with sne
6. 


'''

from train_q2 import ResNet
import torch
from utils import get_data_loader
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
if __name__ == '__main__':
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    path = args.path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(path, map_location=device)
    model.eval()
    model.resnet.fc = nn.Identity()
    batch_size = 16
    test_loader = get_data_loader('voc', train=False, batch_size=batch_size, split='test', inp_size=224)

    N = 0
    pca = PCA(n_components=50)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)

    features = []
    colors = []
    class_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]  # shape (20,3), RGB only

    legend_handles = [
        mpatches.Patch(color=class_colors[i], label=cls)
        for i, cls in enumerate(class_names)
    ]

    # process 1000 images
    with torch.inference_mode():
        for batch_idx, (data, target, wgt) in enumerate(test_loader):
            if N > 1000:
                break
            N += data.shape[0]
            data = data.to(device)
            output = model(data)  # should be b,512,1,1
            output = output.view(-1, 512)
            output = output.detach().cpu().numpy()
            features.append(output)

            # now we need colors
            # target is b,20
            # Use a categorical colormap (e.g., tab20)
            target = target.detach().cpu().numpy()
            # avoid division by zero
            sums = target.sum(axis=1, keepdims=True)  # (b,1)
            sums[sums == 0] = 1
            # weighted average of colors
            point_colors = (target @ class_colors) / sums   # (n,3)
            colors.append(point_colors)
    features = np.vstack(features)
    colors = np.vstack(colors)

    # features should 1000,512
    features = pca.fit_transform(features)  #output should be 1000,50 now
    features = tsne.fit_transform(features)  #output should be 1000,2

#TODO ADD LEGEND
    plt.figure(figsize=(6,6))
    plt.title('t-SNE Visualization')
    plt.scatter(features[:, 0], features[:, 1], c=colors, s=10)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('sne.jpg', dpi=500)
    plt.show()
