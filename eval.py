import torch
from torchviz import make_dot
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Create an instance of the ViTForImageClassification model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Convert the model's parameters to torch.float32
model = model.to(torch.float)

# Create an instance of the ViTFeatureExtractor for preprocessing the input image
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Preprocess the input image
input_image = torch.randn(1, 3, 224, 224)

# Normalize the input image to have values between 0 and 1
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
input_image = normalize(input_image)

# Convert the input image to PIL format
input_image = transforms.ToPILImage()(input_image[0])

# Preprocess the PIL image using the ViTFeatureExtractor
inputs = feature_extractor(images=input_image, return_tensors='pt')

# Convert the input tensor to torch.float32
inputs = {k: v.float() for k, v in inputs.items()}

# Generate a computation graph by passing the input through the model
output = model(**inputs)

# Use torchviz to create a visualization of the computation graph
dot = make_dot(output.logits, params=dict(model.named_parameters()))

# Save the visualization as a PDF file
dot.render("computation_graph", format="pdf")

