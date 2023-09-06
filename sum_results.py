import pandas as pd
import os

files = [
    "/sise/home/barasa/8_bits_attack/experiments/September/lr_0.0020000000949949026_epsilon_0.800000011920929_norm_inf_weights[[1, 50.0, 0.0]]_nameVIT.csv/universal_attack.csv",
    "/sise/home/barasa/8_bits_attack/experiments/September/lr_0.0020000000949949026_epsilon_0.800000011920929_norm_inf_weights[[1, 0.0, 0.0]]_nameVIT.csv/universal_attack.csv",
    "/sise/home/barasa/8_bits_attack/experiments/September/lr_0.0020000000949949026_epsilon_0.800000011920929_norm_inf_weights[[1, 50.0, 0.01]]_nameVIT.csv/universal_attack.csv"]
# files = [s[:-1] if s.endswith("/") else s for s in files]

for file in files:
    dir = file.rsplit('/', 1)[0]
    df = pd.read_csv(file)
    n = len(df["adv_outliers"])
    df.drop(['clean_CPU_time', 'clean_CPU_mem', 'adv_CPU_time', 'adv_CPU_mem', "batch_id", "img_dir"], axis=1,
            inplace=True)
    save_df = pd.DataFrame()
    df = df.sum()
    if df['clean_outliers'] != 0 and df['adv_outliers'] != 0:
        save_df['outliers_diff'] = pd.Series(((df['adv_outliers'] - df['clean_outliers']) / df['clean_outliers']) * 100)
    save_df['time_diff'] = pd.Series(((df['adv_CUDA_time'] - df['clean_CUDA_time']) / df['clean_CUDA_time']) * 100)
    save_df['mem_diff'] = ((df['adv_CUDA_mem'] - df['clean_CUDA_mem']) / df['clean_CUDA_mem']) * 100
    save_df['accuracy'] = df['accuracy'] / n

    path_save = os.path.join(dir, 'results.csv')
    save_df.to_csv(path_save, index=False)
    print(f"saved to {path_save}")
