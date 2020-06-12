import yaml
import os

import trainer
import tester
import data
import transforms

def main(cfgs):
    trans_in_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_train = data.dataset(cfgs, flag='train', trans=trans_in_train)
    trainer.trainer(cfgs, dataset_train)

if __name__ == "__main__":
    with open('./cfgs.yaml') as file_id:
        cfgs = yaml.safe_load(file_id)
    
    cfgs['device'] = 'cuda:0'

    # BSDS and BSDS-VOC
    cfgs['dataset'] = 'NYUD-image'
    cfgs['backbone'] = 'vgg'
    cfgs['loss'] = 'AF'
    cfgs['batch_size'] = 1
    cfgs['acc_grad'] = 10
    cfgs['lr'] = 0.01
    cfgs['max_epoch'] = 32
    cfgs['decay_steps'] = 8

    print(cfgs)
    main(cfgs)