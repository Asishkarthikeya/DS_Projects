import torch
from torch.nn import Conv2d, ReLU, Sequential, Softmax, Upsample, BatchNorm2d, ConvTranspose2d
from color_code import ColorCode

class RCNN(ColorCode):
    def __init__(self, pretrained=False, norm_layer=BatchNorm2d):
        super(RCNN, self).__init__()

        model1=[Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[ReLU(True),]
        model1+=[Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[ReLU(True),]
        model2+=[Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[ReLU(True),]
        model3+=[Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[ReLU(True),]
        model3+=[Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[ReLU(True),]
        model4+=[Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[ReLU(True),]
        model4+=[Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[ReLU(True),]
        model5+=[Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[ReLU(True),]
        model5+=[Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[ReLU(True),]
        model5+=[norm_layer(512),]

        model6= [Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[ReLU(True),]
        model6+=[Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[ReLU(True),]
        model6+=[Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[ReLU(True),]
        model6+=[norm_layer(512),]

        model7= [Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[ReLU(True),]
        model7+=[Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[ReLU(True),]
        model7+=[Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[ReLU(True),]
        model7+=[norm_layer(512),]

        model8= [ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[ReLU(True),]
        model8+=[Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[ReLU(True),]
        model8+=[Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[ReLU(True),]

        model8+=[Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = Sequential(*model1)
        self.model2 = Sequential(*model2)
        self.model3 = Sequential(*model3)
        self.model4 = Sequential(*model4)
        self.model5 = Sequential(*model5)
        self.model6 = Sequential(*model6)
        self.model7 = Sequential(*model7)
        self.model8 = Sequential(*model8)

        self.softmax = Softmax(dim=1)
        self.model_out = Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = Upsample(scale_factor=4, mode='bilinear')
        
    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def load_model_from_github():
    model_url = 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/black_white_colorization_using_faster_rcnn/colorization_release_v2-9b330a0b.pth'
    model = RCNN(pretrained=False)
    model.load_state_dict(torch.load(model_url, map_location='cpu'))
    return model
