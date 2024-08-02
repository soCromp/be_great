from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import os
import pandas as pd

savedir = '/mnt/data/sonia/ckpts/great/adult/moegreatdgpt2-jul29/checkpoint-22575'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# dataset = 'adult'
# if dataset == 'cah':
#     data = fetch_california_housing(as_frame=True).frame
# elif dataset == 'adult':
#     data = pd.read_csv('adult.csv')

# model = GReaT(llm='distilgpt2', batch_size=1,  
#               epochs=1, save_steps=3225,
#               experiment_dir=savedir, multihead=True)
# cols = ['age', 'workclass', 'income']
# # cols = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race',
# #         'sex','capital-gain','capital-loss','hours-per-week','native-country','income']
# model.fit(data[cols])
# model.save(savedir)

model = GReaT.load_from_dir(savedir)
synthetic_data = model.sample(n_samples=10, parse=False, k=1, max_length=20)
synthetic_data = [l[0]+'\n' for l in synthetic_data] #remove [] around batch of 1 sample

with open(os.path.join(savedir, 'samples.csv'), 'w') as f:
    f.writelines(synthetic_data)
