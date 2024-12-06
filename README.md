# On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training

Chen Liu, Zhichao Huang, Mathieu Salzmann, Tong Zhang, Sabine Süsstrunk

Journal of Machine Learning Research (JMLR), Volume 25, 2024

[https://arxiv.org/pdf/2112.07324](https://arxiv.org/pdf/2112.07324)

## Abstract

Adversarial training is a popular method to robustify models against adversarial attacks. However, it exhibits much more severe overfitting than training on clean inputs. In this work, we investigate this phenomenon from the perspective of training instances, i.e., training input-target pairs. Based on a quantitative metric measuring the relative difficulty of an instance in the training set, we analyze the model's behavior on training instances of different difficulty levels. This lets us demonstrate that the decay in generalization performance of adversarial training is a result of fitting hard adversarial instances. We theoretically verify our observations for both linear and general nonlinear models, proving that models trained on hard instances have worse generalization performance than ones trained on easy instances, and that this generalization gap increases with the size of the adversarial budget. Finally, we investigate solutions to mitigate adversarial overfitting in several scenarios, including fast adversarial training and fine-tuning a pretrained model with additional data. Our results demonstrate that using training data adaptively improves the model's robustness.

## Requirements

```
python      >= 3.7
pytorch     >= 1.3
torchvision >= 0.4
autoattack  >= 0.1
```

## Instructions

We rewrite the data loader for the dataset SVHN and CIFAR10 under the folder `dataset`.
This will enable us to use a subset of the training set (Section 4) or to use additional data for training (Section 6).
`util/attack.py` and `util/data_parser.py` define different attack algorithms and data loaders used in this paper.
To load extra data, you should download the corresponding data and put it under `extradata/cifar10`/`extradata/cifar10-c`/`extradata/svhn`. For example, you should download the extra data for cifar10 from [Google Drive](https://drive.google.com/file/d/1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi/view).

`run/train_normal.py` is the main script for training, in which we save the loss value of each instance in each epoch. This information will be used to calculate the instance difficulty.
`run/attack_normal.py` is the main script for evaluation under different attacks. Especially, we use the AutoAttack library ([Github Link](https://github.com/fra31/auto-attack)), the state-of-the-art attacker, for evaluation.
`run/train_fast.py` is the main script for fast adversarial training with adaptive use of easy and hard instances.
`run/tune_normal.py` is the main script for finetuning a pretrained model with additional data.

The scripts to reproduce the key figures in our paper are demonstrated in `Figures.ipynb`. When running the scripts, please download the supporting files from [Google Drive](https://drive.google.com/drive/folders/1sVBzk4xdr1K-_KD_cJSXBLS-vRfcTTHF?usp=sharing) and save them under a folder named `results`.

## Examples

* Adversarially train a RN18 model against PGD on CIFAR10.

```
python run/train_normal.py --valid_ratio 0.02 --epoch_num 100 --model_type resnet --out_folder OUTPUT --model_name 100epoch_resnet --optim name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4 --attack name=pgd,step_size=2,threshold=8,iter_num=10,order=-1 --gpu $GPU_ID$ --lr_schedule name=jump,min_jump_pt=50,jump_freq=25,start_v=0.1,power=0.1
```

* Obtain the difficulty function of PGD adversarial training on a RN18 model.

```
python analyze/sort_instance.py --valid_ratio 0.02 --min_epoch 10 --json_file OUTPUT/100epoch_resnet.json --metric loss --out_file OUTPUT/100epoch_resnet_sortby_loss.json
```

* PGD adversairl training using the easiest/hardest 10000 training instances.

```
python run/train_normal.py --valid_ratio 0.02 --per_file OUTPUT/100epoch_resnet_sortby_loss.json --subset num=10000,mode=easy --epoch_num 200 --out_folder OUTPUT --model_name 200epoch_resnet_easy10k --optim name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4 --lr_schedule name=jump,min_jump_pt=100,jump_freq=50,start_v=0.1,power=0.1 --attack name=pgd,step_size=2,threshold=8,iter_num=10,order=-1 --gpu $GPU_ID$

python run/train_normal.py --valid_ratio 0.02 --per_file OUTPUT/100epoch_resnet_sortby_loss.json --subset num=10000,mode=hard --epoch_num 200 --out_folder OUTPUT --model_name 200epoch_resnet_hard10k --optim name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4 --lr_schedule name=jump,min_jump_pt=100,jump_freq=50,start_v=0.1,power=0.1 --attack name=pgd,step_size=2,threshold=8,iter_num=10,order=-1 --gpu $GPU_ID$
```

* Using ATTA-based fast adversarial training to train a CIFAR10 model using adaptive target or reweighting.

```
python run/train_fast.py --aug_policy crop,vflip --epoch_num 38 --model_type wrn --out_folder OUTPUT/adv_sat_fast --model_name rho0.9_beta0.1 --optim name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4 --lr_schedule name=jump,min_jump_pt=30,jump_freq=6,start_v=0.1,power=0.1 --threshold 8 --step_size 4 --delta_reset 1000 --rho 0.9 --beta 0.1 --warmup 5 --test_attack name=pgd,step_size=2,threshold=8,iter_num=10,order=-1 --gpu $GPU_ID$ --loss_param name=ce

python run/train_fast.py --aug_policy crop,vflip --epoch_num 38 --model_type wrn --out_folder OUTPUT/adv_sat_fast --model_name rho0.9_beta1.0_warmup5_ce_nat-prob_rw --optim name=sgd,lr=1e-1,momentum=0.9,weight_decay=5e-4 --lr_schedule name=jump,min_jump_pt=30,jump_freq=6,start_v=0.1,power=0.1 --threshold 8 --step_size 4 --delta_reset 1000 --rho 0.9 --beta 1.0 --warmup 5 --warmup_rw 5 --test_attack name=pgd,step_size=2,threshold=8,iter_num=10,order=-1 --gpu $GPU_ID$ --loss_param name=ce --batch_size 128 --reweight nat_prob
```

* Finetune a pretrained CIFAR10 model using re-weighting and KL-regularization for 1 epoch. To disable re-weighting, replace the last hyper-parameter with `gamma=6,ada_weight=0`. To disable KL-regularization, replace the last hyper-parameter with `gamma=0`.

```
python run/tune_normal.py --valid_ratio 0.02 --model_type wrn --valid_freq 500 --plus_prop 0.5 --batch_size 128 --epoch_num 1 --model2load PRETRAINED.ckpt --out_folder OUTPUT --model_name kl_reweighed_1epoch --optim name=sgd,lr=1e-3,momentum=0.9,weight_decay=5e-4 --lr_schedule name=const,start_v=0.001 --attack name=pgd,step_size=2,threshold=8,iter_num=10,order=-1 --gpu $GPU_ID$ --tune_mode kl --tune_params gamma=6
```

* Evaluate the robustness of a model by different attacks.

```
python run/attack_normal.py --dataset cifar10 --model_type wrn --model2load OUTPUT/100epoch_resnet.ckpt --out_file OUTPUT/output_apgd100-ce.json --gpu $GPU_ID$ --attack name=apgd,threshold=8,iter_num=100,order=-1,rho=0.75,loss_type=ce

python run/attack_normal.py --dataset cifar10 --model_type wrn --model2load OUTPUT/100epoch_resnet.ckpt --out_file OUTPUT/output_apgd100-dlr.json --gpu $GPU_ID$ --attack name=apgd,threshold=8,iter_num=100,order=-1,rho=0.75,loss_type=dlr

python run/attack_normal.py --dataset cifar10 --model_type wrn --model2load OUTPUT/100epoch_resnet.ckpt --out_file OUTPUT/output_square5000.json --gpu $GPU_ID$ --attack name=square,threshold=8,iter_num=5000,order=-1,window_size_factor=0
```

* Calculate the bound of the Lipschitz constant of the model.

```
python analyze/calc_global_lip.py --iter_num 100 --model2load OUTPUT/100epoch_resnet.ckpt --out_file OUTPUT/output_lipschitz.json --gpu $GPU_ID$
```

## Bibliography

```
@article{liu2024impact,
  author  = {Liu, Chen and Huang, Zhichao and Salzmann, Mathieu and Zhang, Tong and S{\"u}sstrunk, Sabine},
  title   = {On the impact of hard adversarial instances on overfitting in adversarial training},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  pages   = {1--46},
}
```
