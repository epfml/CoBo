# CoBo: Collaborative learning via Bilevel optimization

This is the code base for the paper [CoBo: Collaborative Learning via Bilevel optimization](https://arxiv.org/abs/2409.05539)

## Quickstart

Install dependencies:

```
pip install -r requirements.txt
```

#### Generate dataset manually

If you want to generate datasets before running fine-tuning:

```
python ./src/gen_dataset.py <dataset_name>
```

### Base command

The code consists two parts:
1. Personalized vision models
2. Personalized collaborative learning which is based on the code base for [Personalized Collaborative Fine-tuning](https://github.com/epfml/personalized-collaborative-llms)

   
Here's the base command for running an experiment for collaborative training of language models on Wiki40b:

```
python ./personalized-collaborative-llms/src/main.py --trust cobo --trust_freq 1 \
 --pretraining_rounds 0 --iterations 500 --num_clients 4 --eval_freq 5 \
 --dataset wiki40b --config_format lora --use_pretrained gpt2 --lora_mlp \
 --lora_causal_self_attention --lora_freeze_all_non_lora --no_compile
```

And this is the base command for running an experiment for collaborative training of modles on Cifar100:

```
python ./personalized-vision-models/run.py --bs_train 128 --bs_test 500 --workers 2 2 2 2 \
--gpus 1 --rho 0.05 --lr 0.1 --train_method cobo --iterations 40000 --run_name <run-name>
```

