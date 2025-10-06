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
2. pass through model in batches
load model
'''