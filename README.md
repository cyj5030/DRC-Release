## [Learning Crisp Boundaries Using Deep Refinement Network and Adaptive Weighting Loss]

### Citations

If you are using the code/model/data provided here in a publication, please consider citing our paper:  
  
@article{cao2020learning,  
    author = {Cao, Yi-Jun and Lin, Chuan and Li, Yong-Jie},  
    year = {2020},  
    title = {Learning Crisp Boundaries Using Deep Refinement Network and Adaptive Weighting Loss},  
    journal = {IEEE Transactions on Multimedia},  
    doi = {10.1109/TMM.2020.2987685}  
    }  

### Precomputed results

Evaluation results for BSDS500 and NYUD datasets are available [here](https://drive.google.com/drive/folders/1cjzBpHgEf8nOZZAthGyb3mItRQNDknOu?usp=sharing).

For plot PR-curve or UCM, you can use [here](https://github.com/jponttuset/seism).

### Pretrained models

Pretrained models are available [here](https://drive.google.com/drive/folders/1cjzBpHgEf8nOZZAthGyb3mItRQNDknOu?usp=sharing).

### Testing

1. Clone the repository

2. Download pretrained models, and put them into `$ROOT_DIR/$MODEL_NAME/` folder.

3. Download the datasets you need (you can download from [RCF page](https://github.com/yun-liu/RCF)), and modify the `cfgs.yaml` file.

4. run `test_bsds.py` or `test_nyud.py`.

Note: Before evaluating the predicted edges, you should do the standard non-maximum suppression (NMS) and edge thinning. We used Piotr's Structured Forest matlab toolbox available [here](https://github.com/pdollar/edges).

### Training

1. Download the datasets you need.

2. run `train_bsds.py` or `train_nyud.py`.


### Related Projects

[Richer Convolutional Features for Edge Detection](https://github.com/yun-liu/RCF)