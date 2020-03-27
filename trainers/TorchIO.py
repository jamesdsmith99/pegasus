import numpy as np
import torch


def load_tensor_from_npy(path):
    return torch.from_numpy(np.load(path))

def save_model_optimiser(path, model, optimiser):
    state = {
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict()
    }

    torch.save(state, path)

def load_model(path, model, optimiser=None):
    state = torch.load(path)
    m = model.load_state_dict(state['model'])
    if optimiser:
        o = optimiser.load_state_dict(state['optimiser'])
    return [m, o if optimiser else 'didnt load optimiser state']
