import torch
from src.models.layers import inconv, down, up, outconv
import torch.nn.functional as F
import os
from src.models.base_model import BaseModel
from src.enums import DictKeys

#TODO remove save feature map

class UNet(BaseModel):
    """
    Vanilla unet model architecture
    """
    def __init__(self, n_classes, batchnorm=False):
        super(UNet, self).__init__()
        self.inc = inconv(3, 64, batchnorm)
        self.down1 = down(64, 128, batchnorm)
        self.down2 = down(128, 256, batchnorm)
        self.down3 = down(256, 512, batchnorm)
        self.down4 = down(512, 512, batchnorm)
        self.up1 = up(1024, 256, batchnorm)
        self.up2 = up(512, 128, batchnorm)
        self.up3 = up(256, 64, batchnorm)
        self.up4 = up(128, 64, batchnorm)
        self.outc = outconv(64, n_classes)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    def forward_step(self, **kwargs):
        x=self.unpack_kwargs(**kwargs)
        out=self.forward(x)
        # out=torch.argmax()
        return {
            DictKeys.Y_PRED.value:out
            }
    def unpack_kwargs(self, **kwargs):
        return kwargs[DictKeys.INPUT.value]
class ConditionalGuidedUNetOutMaskOutClass(BaseModel):
    """
    Vanilla unet model architecture
    """
    def __init__(self, n_classes,n_organs, batchnorm=False):
        super(ConditionalGuidedUNetOutMaskOutClass, self).__init__()
        self.inc = inconv(3, 64, batchnorm)
        self.down1 = down(64, 128, batchnorm)
        self.down2 = down(128, 256, batchnorm)
        self.down3 = down(256, 512, batchnorm)
        self.down4 = down(512, 512, batchnorm)
        self.up1 = up(1024, 256, batchnorm)
        self.up2 = up(512, 128, batchnorm)
        self.up3 = up(256, 64, batchnorm)
        self.up4 = up(128, 64, batchnorm)
        self.outc = outconv(64, n_classes)
        self.pool=torch.nn.AdaptiveAvgPool2d(1)
        self.fc1=torch.nn.Linear(64,64)
        self.fc2=torch.nn.Linear(64,64)
        self.outorgan=torch.nn.Linear(64,n_organs)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pooled=self.pool(x)
        pooled=torch.flatten(pooled,start_dim=1)
        # print(x.shape)
        x = self.outc(x)
        # print(pooled.shape)
        pooled=self.fc2(self.fc1(pooled))
        
        organ=self.outorgan(pooled)
        return x,organ
    def forward_step(self, **kwargs):
        x=self.unpack_kwargs(**kwargs)
        out,organ=self.forward(x)
        # out=torch.argmax()
        return {
            DictKeys.Y_PRED.value:{
                'mask':out,
                'organ':organ
            }
            }
    def unpack_kwargs(self, **kwargs):
        return kwargs[DictKeys.INPUT.value]
class ConditionalGuidedUNetOutMaskInClass(BaseModel):
    """
    Vanilla unet model architecture
    """
    def __init__(self, n_classes,n_organs, batchnorm=False):
        super(ConditionalGuidedUNetOutMaskInClass, self).__init__()
        self.inc = inconv(3+n_organs, 64, batchnorm)
        self.down1 = down(64, 128, batchnorm)
        self.down2 = down(128, 256, batchnorm)
        self.down3 = down(256, 512, batchnorm)
        self.down4 = down(512, 512, batchnorm)
        self.up1 = up(1024, 256, batchnorm)
        self.up2 = up(512, 128, batchnorm)
        self.up3 = up(256, 64, batchnorm)
        self.up4 = up(128, 64, batchnorm)
        self.outc = outconv(64, n_classes)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x=self.outc(x)
        return x
    def forward_step(self, **kwargs):
        x=self.unpack_kwargs(**kwargs)
        out=self.forward(x)
        # out=torch.argmax()
        return {
            DictKeys.Y_PRED.value:out
            }
    def unpack_kwargs(self, **kwargs):
        return kwargs[DictKeys.INPUT.value]
class UNetLight(BaseModel):
    """
    Vanilla unet model architecture but with less parameters
    """
    def __init__(self, n_classes, batchnorm=False):
        super(UNetLight, self).__init__()
        self.inc = inconv(3, 8, batchnorm)
        self.down1 = down(8, 16, batchnorm)
        self.down2 = down(16, 32, batchnorm)
        self.down3 = down(32, 64, batchnorm)
        self.down4 = down(64, 64, batchnorm)
        self.up1 = up(128, 32, batchnorm)
        self.up2 = up(64, 16, batchnorm)
        self.up3 = up(32, 8, batchnorm)
        self.up4 = up(16, 8, batchnorm)
        self.outc = outconv(8, n_classes)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def forward_step(self, **kwargs):
        x=self.unpack_kwargs(**kwargs)
        out=self.forward(x)
        return {
            DictKeys.Y_PRED.value:out
            }
    def unpack_kwargs(self, **kwargs):
        return kwargs[DictKeys.INPUT.value]