import os
from tqdm import tqdm

from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from art.defences.preprocessor import (SpatialSmoothing, FeatureSqueezing, PixelDefend,
                                       TotalVarMin, ThermometerEncoding, GaussianAugmentation,
                                       LabelSmoothing)
from art.defences.postprocessor import GaussianNoise,ClassLabels,HighConfidence,ReverseSigmoid
from load_ptq4vit import get_ptq4vit_net
import torch.nn.functional as F
from load_RepQ_ViT import get_RepQ_ViT_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# images_path = "/content/inf_0.06274509803921569_0.002_True_500_1_1_ptq4vit_[[1, 0, 0]]_70"
# images_path = "/content/PGD"
# images_path = "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/June/inf_0.06274509803921569_0.002_True_500_1_1_VIT_[[1, 0, 0]]_70/"
# images_path = "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/June/inf_0.06274509803921569_0.002_True_500_1_1_VIT_[[1, 0, 0]]_70/"
# images_path = "/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/June/PGD on llm_int8/"
LLM_int = ["/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/experiments/June/inf_0.06274509803921569_0.002_True_6000_4_1_VIT_[[1, 0.0, 0]]_70/"]
PTQ4VIT_paths = ["/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/ptq4vit_Quant/experiments/July/inf_0.06274509803921569_0.002_True_3000_1_1_ptq4vit_[[1, 0, 0]]_7000/"]
REP_Q = ["/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/RepQ_Quant/experiments/July/inf_0.06274509803921569_0.002_True_3000_1_1_RepQ_[[1, 0, 0]]_7000/"]
No_Quant = ["/dt/shabtaia/dt-fujitsu/8_bit_attack/Second_submission/ptq4vit_No_Quant_VIT/experiments/July/inf_0.06274509803921569_0.002_True_100_1_1_ptq4vit_[[1, 0, 0]]_700/"]
P  = No_Quant
if "ptq4vit" in P[0]:
    if "No_Quant" in P[0]:
        model = get_ptq4vit_net(Quant=False)
    else:
        model = get_ptq4vit_net(m="deit")

elif "RepQ" in P[0]:
    model = get_RepQ_ViT_net()
else:
    model_name = "facebook/deit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name, device_map="auto",
                                                      load_in_8bit=True)

for i,images_path in enumerate(P):
    if i == 1:
        name = "Regular PGD"
    else:
        name = "MY PGD"




    # model = get_ptq4vit_net()
    # Lists to store the filenames
    clean_files = []
    adv_files = []

    acc = []
    acc_defense = []
    for filename in os.listdir(images_path):
        if filename.endswith('clean.pt'):
            clean_files.append(os.path.join(images_path, filename))
        elif filename.endswith('adv.pt'):
            adv_files.append(os.path.join(images_path, filename))

    adv_files = sorted(adv_files)
    clean_files = sorted(clean_files)[:]
    preprocessor = None
    postprocessor = None
    for i, file in enumerate(tqdm(clean_files)):
        # preprocessor = SpatialSmoothing(window_size=3)
        # preprocessor = FeatureSqueezing(clip_values=(0, 1), bit_depth=1)
        # preprocessor = TotalVarMin(clip_values=(0, 1))
        # preprocessor = ThermometerEncoding(clip_values=(0, 1))
        # preprocessor = GaussianAugmentation(sigma=1, augmentation=False) # work


        # postprocessor = GaussianNoise(scale=1)
        # postprocessor = ClassLabels()
        # postprocessor = HighConfidence(cutoff=0.1)
        # postprocessor = ReverseSigmoid(beta=1.0, gamma=0.1)

        file_clean = torch.load(clean_files[i], map_location=device)
        file_adv = torch.load(adv_files[i], map_location=device)

        # for random pert
        max_value = file_clean.max()
        min_value = file_clean.min()

        # Generate a random image with values sampled from a normal distribution
        mean = 0.5 * (max_value + min_value)  # Set the mean to be in the middle of the range
        std_dev = (max_value - min_value) / 4  # Adjust the standard deviation as needed
        random_image = torch.randn(file_adv.size()).to("cuda") * std_dev + mean
        file_adv = file_clean + random_image

        file_clean_defense = file_clean
        file_adv_defense = file_adv

        if preprocessor:
            file_clean_defense = torch.tensor(preprocessor(torch.load(clean_files[i], map_location=device).detach().cpu().numpy())[0]).to(device) # with defense
            file_adv_defense = torch.tensor(preprocessor(torch.load(adv_files[i], map_location=device).detach().cpu().numpy())[0]).to(device) # with defense



        # for model accuracy
        try:
            # for model accuracy
            real_pred = model(file_clean).logits.argmax(-1)
            attack_pred = model(file_adv).logits.argmax(-1)
            acc.append(real_pred.eq(attack_pred).float().mean().item())


            # for model accuracy with defense
            if postprocessor:

                preds = F.softmax(model(attack_pred).logits, dim=-1)
                attack_pred_d = preprocessor(preds).argmax(-1)
                acc_defense.append(real_pred.eq(attack_pred_d).float().mean().item())

            else:
                attack_pred_d = model(file_adv_defense).logits.argmax(-1)
                acc_defense.append(real_pred.eq(attack_pred_d).float().mean().item())

        except AttributeError:

            # for model accuracy
            real_pred = model(file_clean).argmax(-1)
            attack_pred = model(file_adv).argmax(-1)
            acc.append(real_pred.eq(attack_pred).float().mean().item())

           # for model accuracy with defense
            attack_pred_d = model(file_adv_defense).argmax(-1)
            acc_defense.append(real_pred.eq(attack_pred_d).float().mean().item())
        if i % 100 == 0:
            print()
            print(f"{name},Accuracy: ", sum(acc) / len(acc))
            print(f"{name} Accuracy_defense: ", sum(acc_defense) / len(acc_defense))


    print()
    print(f"{name} Accuracy NO defense: ", sum(acc) / len(acc))
    print(f"{name} Accuracy_defense: ", sum(acc_defense) / len(acc_defense))