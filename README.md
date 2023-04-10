# A Cheaper and Better Diffusion Language Model with Soft-Masked Noise


This is the official implementation of the paper: A Cheaper and Better Diffusion Language Model with Soft-Masked Noise.

-----------------------------------------------------
## Usage
One needs to setup the enrironment before running the experiments.


## Conda Setup:
```python 
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb
```

-----------------------------------------------------
## Train Masked-Diffusion-LM:


```cd improved-diffusion; mkdir diffusion_models;```

```python scripts/run_train.py --diff_steps 500 --model_arch bert --lr 0.0003 --lr_anneal_steps 400000  --seed 0 --noise_schedule sqrt  --in_channel 128 --modality roc --submit no --padding_mode pad  --app "--predict_xstart True --training_mode masked-diffuse-lm  --roc_train ../datasets/ROCstory "  --bsz 64```



-------------------
## Decode Diffusion-LM:
mkdir generation_outputs 

``python scripts/batch_decode.py {path-to-diffusion-lm} -1.0 ema``


------------------- 
## Controllable Text Generation 
First, train the classsifier used to guide the generation (e.g. a syntactic parser) 

``  
python train_run.py --experiment e2e-tgt-tree  --app "--init_emb {path-to-diffusion-lm} --n_embd {16} --learned_emb yes " --pretrained_model bert-base-uncased --epoch 6 --bsz 10
``

Then, we can use the trained classifier to guide generation. 
(currently, need to update the classifier directory in scripts/infill.py. I will clean this up in the next release.)

``python 
python scripts/infill.py --model_path {path-to-diffusion-lm} --eval_task_ 'control_tree' --use_ddim True  --notes "tree_adagrad" --eta 1. --verbose pipe``



## Acknowledgement
Part of our codes are adapted from [Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) and [Transformers](https://github.com/huggingface/transformers).


## License
This project is licensed under the Apache-2.0 License.


