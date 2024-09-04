import torch.nn as nn
import torchvision
from transformers import AutoImageProcessor, ViTForImageClassification, BeitForImageClassification, \
    SwinForImageClassification

from transformers import AutoFeatureExtractor, WhisperForAudioClassification, BitsAndBytesConfig
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from load_ptq4vit import get_ptq4vit_net
from load_RepQ_ViT import get_RepQ_ViT_net

def get_model(cfg, model_name):
    model = None
    if model_name == 'VIT':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto",load_in_8bit=True,llm_int8_threshold = 6.0)  # for large model, replace "base" with "large" -> google/vit-large-patch16-224
        # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    elif model_name == 'VIT_tiny':
        model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224", device_map="auto",load_in_8bit=True,llm_int8_threshold = 6.0)
    elif model_name == 'VIT_small':
        model = ViTForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224", device_map="auto",load_in_8bit=True,llm_int8_threshold = 6.0)
    elif model_name == 'VIT_large':
        model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', device_map="auto",load_in_8bit=True,llm_int8_threshold = 6.0)
    elif model_name == 'DeiT':
        model = ViTForImageClassification.from_pretrained("facebook/deit-base-patch16-224",
                                                           device_map="auto",
                                                           load_in_8bit=True)
    elif model_name == 'Whisper':
        model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id", device_map="auto",
                                                              load_in_8bit=True) # "openai/whisper-tiny"

    elif model_name == 'Owldetection':
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16", device_map="auto",
                                                         load_in_8bit=True)
    elif model_name == 'Detr':
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm", device_map='auto', load_in_8bit=True)

    elif model_name == 'yolos':
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small', device_map="auto",
                                                        load_in_8bit=True)
    elif model_name == 'yolos_base':
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-base', device_map="auto",
                                                        load_in_8bit=True)

    elif model_name == 'gpt2':
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", device_map="auto",
                                                         load_in_8bit=True)
    elif model_name == 'blip':
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", device_map="auto",
                                                     load_in_8bit=True)

    elif model_name == 'git':
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco", device_map="auto",
                                                     load_in_8bit=True)
    elif model_name == 'VIT_large':
        model = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384', device_map="auto",
                                                          load_in_8bit=True)
    elif model_name == 'VIT_384':
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch32-384', device_map="auto",
                                                      load_in_8bit=True)

    elif model_name == 'DeiT_large':
        model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-384', device_map="auto", load_in_8bit=True)

    elif model_name == 'BEiT_large':
        model = BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224", device_map="auto", load_in_8bit=True)
    elif model_name == 'BEiT_base':
        model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224", device_map="auto", load_in_8bit=True)
    elif model_name == 'swin_tiny':
        model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", device_map="auto",
                                                   load_in_8bit=True)
    elif model_name == 'swin_base':
        model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224", device_map="auto",
                                                   load_in_8bit=True)
    elif model_name == 'ptq4vit':
        model = get_ptq4vit_net()
    elif model_name == 'RepQ':
        model = get_RepQ_ViT_net()




    # elif model_name == 'other':
    #     model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16", device_map="auto",
    #                                                      load_in_8bit=True)
    # model.eval()
    return model


def get_model_feature_extractor(model_name):
    feature_extractor = None
    dct = {}
    # dct = {"image_mean": [0.485, 0.456, 0.406],
    #          "imag`e_std": [0.229, 0.224, 0.225]}
    if model_name in ['VIT',"ptq4vit","RepQ"]:
        feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', **dct)
    elif model_name == 'VIT_tiny':
        feature_extractor = AutoImageProcessor.from_pretrained('WinKawaks/vit-tiny-patch16-224')
    elif model_name == 'VIT_small':
        feature_extractor = AutoImageProcessor.from_pretrained('WinKawaks/vit-small-patch16-224')
    elif model_name == 'VIT_large':
        feature_extractor = AutoImageProcessor.from_pretrained('google/vit-large-patch16-224')

    elif model_name == 'DeiT':
        feature_extractor = AutoImageProcessor.from_pretrained("facebook/deit-base-patch16-224", **dct)

    elif model_name == 'Whisper':
        feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id", **dct) # openai/whisper-tiny

    elif model_name == 'Owldetection':
        feature_extractor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
    elif model_name == 'Detr':
        feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    elif model_name == 'yolos':
        feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
    elif model_name == 'yolos_base':
        feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
    elif model_name == 'gpt2':
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    elif model_name == 'blip':
        feature_extractor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    elif model_name == 'git':
        feature_extractor = AutoProcessor.from_pretrained("microsoft/git-base-coco")

    elif model_name == 'VIT_large':
        feature_extractor = AutoImageProcessor.from_pretrained('google/vit-large-patch32-384', **dct)
    elif model_name == 'VIT_384':
        feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch32-384', **dct)
    elif model_name == 'DeiT_large':
        feature_extractor = AutoImageProcessor.from_pretrained('facebook/deit-base-patch16-384', **dct)
    elif model_name == 'BEiT_large':
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/beit-large-patch16-224", **dct)
    elif model_name == 'BEiT_base':
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224", **dct)
    elif model_name == 'swin_tiny':
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", **dct)
    elif model_name == 'swin_base':
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224", **dct)

    return feature_extractor



