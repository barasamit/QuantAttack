
import torch.nn as nn
import torchvision
from transformers import AutoImageProcessor, ViTForImageClassification


def get_vit_model(cfg):

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto",
                                                      load_in_8bit=True)
    return model


def get_vit_feature_extractor():
    feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    return feature_extractor


def get_classification_model(estimator_config):
    if estimator_config['model_arch'] == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=estimator_config['num_of_classes'])
        )
        print("Load Resnet18 pretrained model")


def freeze_bn_stats(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()


def unfreeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.train()