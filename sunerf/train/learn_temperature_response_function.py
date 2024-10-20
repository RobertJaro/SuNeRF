import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from sunerf.model.model import GenericModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--channels', type=int, nargs='+', required=False, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    temperature_response_function = np.load(args.response_file)
    temperature = temperature_response_function['temperature']
    response = temperature_response_function['response']
    response = response[args.channels] if args.channels is not None else response

    # normalize data
    response = response / np.max(response) / 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GenericModel(1, response.shape[0], dim=16, n_layers=2)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    temperature_tensor = torch.tensor(np.log10(temperature)).float().to(device)[:, None]
    response_tensor = torch.tensor(np.log10(response)).float().to(device).T
    test_temperature = torch.linspace(temperature_tensor.min(), temperature_tensor.max(), 200).to(device)[:, None]

    for i in range(int(1e5)):
        model.train()
        optimizer.zero_grad()
        output = model(temperature_tensor)
        loss = criterion(output, response_tensor)
        loss.backward()
        optimizer.step()
        if i % 1e4 == 0:
            print(f'Epoch {i}: Loss {loss.item()}')
            with torch.no_grad():
                model.eval()
                test_output = model(test_temperature)
                fig, axs = plt.subplots(response.shape[0], 1, figsize=(5, 15))
                axs = [axs] if response.shape[0] == 1 else axs
                for i, ax in enumerate(axs):
                    ax.plot(temperature, response[i], 'o', color='black')
                    ax.plot((10 ** test_temperature).cpu().numpy(), (10 ** test_output).cpu().numpy()[:, i],
                            color='red')
                    ax.loglog()
                fig.tight_layout()
                plt.savefig(args.out_file.replace('.pt', '.jpg'))
                plt.close()
            torch.save(model, args.out_file)
    torch.save(model, args.out_file)
