import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

import torchutils
from modular_cnn import ModularCNN, make_layers
from vis_utils import preprocess_image, recreate_image, save_image


class CNNLayerVisualization:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        torchutils.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (1, 10, 100, 100)))
        # Process image and return variable
        processed_image = preprocess_image(random_image)
        print(processed_image.shape)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.5, weight_decay=1e-6)

        for i in range(1, 101):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.6f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 10 == 0:
                im_path = '../generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)


if __name__ == '__main__':
    hyper_params = {"adaptive_pool": (5, 7, 7)
               ,"features": [8,8,"M",8,8,"M",32,32,32,"M",64,64,"M"]
               ,"classifier": [0.6, 400, 0.6, 200]
             }

    cnn_layer = 28
    for i in range(10):
        filter_pos = i

        from collections import OrderedDict

        state_dict = torch.load("/Users/idofarhi/Documents/Thesis/Code/model.pt", map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v

        pretrained_model = ModularCNN(make_layers(hyper_params["features"], batch_norm=True), classifier = hyper_params["classifier"], adaptive_pool=hyper_params["adaptive_pool"])
        pretrained_model.load_state_dict(new_state_dict)
        pretrained_model = pretrained_model.features

        layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

        # Layer visualization without pytorch hooks
        layer_vis.visualise_layer_without_hooks()