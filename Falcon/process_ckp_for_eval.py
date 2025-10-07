# ccoding=utf-8
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="the trained model")
parser.add_argument("output_model_path", help="homie observes 29 instead of 27 dof")

if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_path_v0 = args.model_path
    pretrained_state = torch.load(checkpoint_path_v0, map_location=torch.device('cpu'))
    model_state_dict = pretrained_state[0]['state_dict']

    filtered_pretrained_state_dict = {
        k: v for k, v in pretrained_state[0]['state_dict'].items()
        if not k.startswith('aux_loss_modules')
    }

    pretrained_state[0]['state_dict'] = filtered_pretrained_state_dict
    torch.save(pretrained_state, args.output_model_path)
