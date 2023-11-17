import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, pdrop=0, ker=3, final_relu=True, fac=64):
        """A variation of the UNet (1505.04597) including dropout and BatchNorm.

        pdrop: dropout rate for dropout layers
        ker: kernel size (default 3 was best in a previous problem)
        final_relu: bool whether to include a final relu
        fac: multiplicative factor on the number of channels (default 64)
        """

        super().__init__()
        self.pdrop = pdrop
        self.ker = ker
        self.final_relu = final_relu

        # Encode
        self.conv_encode1 = self.standard_block(in_channels, fac, fac)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.standard_block(fac, 2*fac, 2*fac)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.standard_block(2*fac, 4*fac, 4*fac)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = self.standard_block(4*fac, 8*fac, 4*fac)

        # Decode
        self.upsample3 = nn.ConvTranspose2d(in_channels=4*fac,
                                            out_channels=4*fac,
                                            kernel_size=2
                                            stride=2)
        self.conv_decode3 = self.standard_block(8*fac, 4*fac, 2*fac)
        self.upsample2 = nn.ConvTranspose2d(in_channels=2*fac,
                                            out_channels=2*fac,
                                            kernel_size=2
                                            stride=2)
        self.conv_decode2 = self.standard_block(4*fac, 2*fac, fac)
        self.upsample1 = nn.ConvTranspose2d(in_channels=fac,
                                            out_channels=fac,
                                            kernel_size=2
                                            stride=2)
        self.final_layer = self.final_block(2*fac, fac, out_channels)
    
    def standard_block(self, in_channels, mid_channels, out_channels):
        """This is the standard block that the UNet is composed of. It consists of 2
        cycles of Convolution -> ReLU -> Dropout -> Batch Norm.
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=self.ker,
                      in_channels=in_channels,
                      out_channels=mid_channels,
                      padding=self.ker//2),
            ## padding of floor(kernel_size/2) maintains height and width
            nn.ReLU(),
            nn.Dropout2d(p=self.pdrop),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=self.ker,
                      in_channels=mid_channels,
                      out_channels=out_channels,
                      padding=self.ker//2),
            nn.ReLU(),
            nn.Dropout2d(p=self.pdrop),
            nn.BatchNorm2d()
        )
        return block

    def final_block(self, in_channels, mid_channels, out_channels):
        """The final block is a lot like the others but has an extra convolution and
        possibly ReLU at the end. (In addition, a further sigmoid is required to
        interpret the outputs as probabilities, but that is taken care of in the
        loss function during training.)

        """

        std_block = self.standard_block(in_channels, mid_channels, mid_channels)

        final_conv = nn.Conv2d(kernel_size=self.ker,
                               in_channels=mid_channels,
                               out_channels=out_channels,
                               padding=self.ker//2)
        if self.final_relu:
            block = nn.Sequential(
                *std_block,
                final_conv,
                nn.ReLU()
            )
        else:
            block = nn.Sequential(
                *std_block,
                final_conv
            )
            
        return block
        

    def forward(self, x):
        """Run the UNet on an input x"""

        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)

        # Decode
        upsampled3 = self.upsample3(bottleneck1)
        up_and_cat1 = torch.cat((upsampled3, encode_block3), 1)

        decode2 = self.conv_decode3(up_and_cat1)
        upsampled2 = self.upsample2(decode2)

        up_and_cat2 = torch.cat((upsampled2, encode_block2), 1)

        decode1 = self.conv_decode2(up_and_cat2)
        upsampled1 = self.upsample1(decode1)
        up_and_cat1 = torch.cat((upsampled1, encode_block1), 1)
        out = self.final_layer(up_and_cat1)

        return out

