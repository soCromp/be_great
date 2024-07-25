from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import os
import pandas as pd

savedir = 'ckpts/great/adult/moe/dgpt2/jul24/checkpoint-6450/model.pt'
if not os.path.exists(savedir):
    os.makedirs(savedir)

dataset = 'adult'
if dataset == 'cah':
    data = fetch_california_housing(as_frame=True).frame
elif dataset == 'adult':
    data = pd.read_csv('/hdd3/sonia/data/adult.csv')

model = GReaT(llm='distilgpt2', batch_size=1,  
              epochs=1, save_steps=3225,
              experiment_dir=savedir, multihead=True)
# model.fit(data)
# model.save(savedir)

model.load_finetuned_model(savedir)
# synthetic_data = model.sample(n_samples=10000)
# synthetic_data.to_csv(os.path.join(savedir, 'samples.csv'), index=False)
