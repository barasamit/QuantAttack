
from transformers import AutoImageProcessor, ViTForImageClassification


def get_vit_model(cfg):

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map="auto",
                                                      load_in_8bit=True,load_in_8bit_threshold=cfg.model_threshold)

    return model


def get_vit_feature_extractor():
    feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    return feature_extractor



