from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import os
import pandas as pd

savedir = '/mnt/data/sonia/ckpts/great/adult/dgpt2-jul21'
# if not os.path.exists(savedir):
#     os.makedirs(savedir)

# dataset = 'adult'
# if dataset == 'cah':
#     data = fetch_california_housing(as_frame=True).frame
# elif dataset == 'adult':
#     data = pd.read_csv('/mnt/data/sonia/datasets/adult/adult.csv')

# model = GReaT(llm='distilgpt2', batch_size=16,  
#               epochs=50, 
#               save_steps=3225, experiment_dir=savedir)
# model.fit(data)
# model.save(savedir)

model = GReaT.load_from_dir(savedir)
synthetic_data = model.sample(n_samples=10000)
synthetic_data.to_csv(os.path.join(savedir, 'samples.csv'), index=False)
