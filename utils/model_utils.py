import torch.nn as nn
import torchvision
from transformers import AutoImageProcessor, ViTForImageClassification, ViTFeatureExtractor, DeiTForMaskedImageModeling


def get_model(cfg,model_name):
    if model_name == 'VIT':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto",
                                                          load_in_8bit=True, load_in_8bit_threshold=cfg.model_threshold)
    elif model_name == 'DeiT':
        model = DeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224", device_map="auto",
                                                          load_in_8bit=True, load_in_8bit_threshold=cfg.model_threshold)
    return model


def get_model_feature_extractor(model_name):
    if model_name == 'VIT':
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    elif model_name == 'DeiT':
        feature_extractor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    return feature_extractor


def get_classification_model(estimator_config):
    if estimator_config['model_arch'] == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=estimator_config['num_of_classes'])
        )
        print("Load Resnet18 pretrained model")
