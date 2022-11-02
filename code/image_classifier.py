import torch.nn as nn
import timm

class ImageClassifier(nn.Module):
    def __init__(self, drop_p:float, base_model_ckpt: str):
        super(ImageClassifier, self).__init__()

        if base_model_ckpt != None:
            self.base_model = timm.create_model("efficientnet_b2", pretrained = False, checkpoint_path=base_model_ckpt)
        else:
            self.base_model = timm.create_model('efficientnet_b2', pretrained=True)

        self.dropout = nn.Dropout(drop_p)

        self.activation_function = nn.GELU()
        self.classifier = nn.Linear(1000, 3)
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        base_output = self.base_model(input)

        pooler = self.dropout(base_output)
        pooler = self.activation_function(pooler)

        classifier_output = self.classifier(pooler)
        # softmax_result = self.softmax(classifier_output)

        return classifier_output