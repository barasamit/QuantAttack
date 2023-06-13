import torch.nn as nn
import torchvision
from transformers import AutoImageProcessor, ViTForImageClassification, ViTFeatureExtractor, DeiTForMaskedImageModeling, \
    RegNetForImageClassification, BeitForImageClassification, DetrForObjectDetection, ViTForMaskedImageModeling, \
    ViTMAEModel, ViTMSNForImageClassification, BeitForSemanticSegmentation, WhisperForAudioClassification, \
    FlaxViTForImageClassification, SwinForImageClassification, Swinv2ForImageClassification, \
    ConvNextForImageClassification, LevitForImageClassification, ImageGPTForImageClassification, \
    EfficientFormerForImageClassification, DPTForSemanticSegmentation, DeiTForImageClassificationWithTeacher

from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import AutoProcessor, CLIPSegForImageSegmentation


def get_model(cfg, model_name):
    model = None
    if model_name == 'VIT':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto",
                                                          load_in_8bit=True)
    elif model_name == 'DeiT':
        model = DeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224",
                                                           device_map="auto",
                                                           load_in_8bit=True)
    elif model_name == 'Whisper':
        model = WhisperForAudioClassification.from_pretrained("openai/whisper-tiny", device_map="auto",
                                                              load_in_8bit=True)

    elif model_name == 'Owldetection':
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map="auto",
                                                         load_in_8bit=True)

    elif model_name == 'other':
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", device_map="auto",
                                                         load_in_8bit=True)
    return model


def get_model_feature_extractor(model_name):
    feature_extractor = None

    if model_name == 'VIT':
        feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

    elif model_name == 'DeiT':
        feature_extractor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")

    elif model_name == 'Whisper':
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")

    elif model_name == 'Owldetection':
        feature_extractor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    elif model_name == 'other':
        feature_extractor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    return feature_extractor


def get_classification_model(estimator_config):
    if estimator_config['model_arch'] == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=estimator_config['num_of_classes'])
        )
        print("Load Resnet18 pretrained model")
