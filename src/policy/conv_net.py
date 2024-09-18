import torch
import torch.nn as nn

# conv_layers_params = [
#     {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
#     {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'},
#     {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'activation': 'relu'}
# ]

# # Create an instance of the CustomConvNet
# model = CustomConvNet(
#     input_channels=3,        # For RGB images
#     input_height=32,         # Height of the input images
#     input_width=32,          # Width of the input images
#     conv_layers_params=conv_layers_params,
#     output_size=10           # For 10 output classes, for example
# )

class CustomConvNet(nn.Module):
    def __init__(self, input_channels, input_height, input_width, conv_layers_params, output_size):
        super(CustomConvNet, self).__init__()
        # Save input dimensions
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        layers = []
        in_channels = input_channels

        # Build convolutional layers based on the provided parameters
        for layer_params in conv_layers_params:
            out_channels = layer_params.get('out_channels')
            kernel_size = layer_params.get('kernel_size', 3)
            stride = layer_params.get('stride', 1)
            padding = layer_params.get('padding', 0)
            activation = layer_params.get('activation', 'relu')

            # Add convolutional layer
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            layers.append(conv_layer)

            # Add activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            # You can add more activation functions as needed

            # Update in_channels for the next layer
            in_channels = out_channels

        # Combine all layers into a sequential module
        self.conv = nn.Sequential(*layers)

        # Compute the output size after convolutional layers to define the fully connected layer
        self.fc_input_size = self._get_conv_output_size()
        self.fc = nn.Linear(self.fc_input_size, output_size)

    def _get_conv_output_size(self):
        # Create a dummy input tensor with the same size as the input images
        input_size = (1, self.input_channels, self.input_height, self.input_width)
        dummy_input = torch.zeros(input_size)
        # Pass the dummy input through the convolutional layers
        output_feat = self.conv(dummy_input)
        # Flatten the output and get its size
        n_size = output_feat.view(1, -1).size(1)
        return n_size

    def forward(self, x):
        # Pass input through convolutional layers
        x = self.conv(x)
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        # Pass through the fully connected layer
        x = self.fc(x)
        return x
