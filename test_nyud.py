import yaml
import os

import trainer
import tester
import data
import transforms

def main(cfgs):
    trans_in_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_test = data.dataset(cfgs, flag='test', trans=trans_in_test)
    tester.tester(cfgs, dataset_test)
    

if __name__ == "__main__":
    with open('./cfgs.yaml') as file_id:
        cfgs = yaml.safe_load(file_id)
    
    cfgs['device'] = 'cuda:0'

    # BSDS and BSDS-VOC
    cfgs['dataset'] = 'NYUD-image'  # BSDS  BSDS-VOC
    cfgs['backbone'] = 'resnet50' # vgg resnet50 ...
    print(cfgs)
    main(cfgs)