import os
import pandas as pd
from tqdm import tqdm
results_combine = pd.DataFrame()
main_dir = "/sise/home/barasa/8_bit/results/"
save_dir = "/sise/home/barasa/8_bit/results/stats"
csv_files = [f for f in os.listdir(main_dir) if f.endswith('.csv')]

for csv_file in tqdm(csv_files):
    open_file = os.path.join(main_dir,csv_file)
    df = pd.read_csv(open_file)
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    stats = pd.DataFrame({'mean': df[numeric_cols].mean(), 'std': df[numeric_cols].std()}).transpose()
    multi_index = pd.MultiIndex.from_product([[csv_file], ['mean', 'std']])
    stats.index = multi_index
    save_file = os.path.join(save_dir,csv_file)
    stats_file = save_file.rsplit(".",1)[0] + '_stats.csv'
    results_combine = pd.concat([results_combine, stats], axis=0)
results_combine.to_csv(stats_file)