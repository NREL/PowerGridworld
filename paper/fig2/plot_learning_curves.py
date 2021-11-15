import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

PLOT_FONT_SIZE = 12
MAX_ITER = 350


def plot_learning_curves(data_dir, exp_name):

    items = ['rewards', 'agrewards', 'ploss', 'qloss', 'vvio']
    logged_data = {x: [] for x in items}

    for item in items:
        try:
            file_name = os.path.join(data_dir, exp_name + '_' + item + '.pkl')
            with open(file_name, 'rb') as fp:
                logged_data[item] = pickle.load(fp)
        except FileNotFoundError:
            continue

    # Imply the agent number
    agent_num = int(
        len(logged_data['agrewards']) / len(logged_data['rewards']))

    logged_data['agrewards'] = np.array(logged_data['agrewards'])
    logged_data['agrewards'] = logged_data['agrewards'].reshape(
        (-1, agent_num))

    fig, axes = plt.subplots(3, 2)

    for agent_idx in range(agent_num):
        axes[0, 0].plot(-1 * logged_data['agrewards'][:, agent_idx][:MAX_ITER],
                     label='Agent ' + str(agent_idx + 1))
    # axes[0, 0].legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.9),)
    axes[0, 0].grid()
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Agent Cost', fontsize=PLOT_FONT_SIZE)

    axes[0, 1].plot(-1 * np.array(logged_data['rewards'])[:MAX_ITER],
                    color='k', alpha=0.8)
    axes[0, 1].grid()
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylabel('Total Cost', fontsize=PLOT_FONT_SIZE)
    axes[0, 1].yaxis.set_label_position("right")
    axes[0, 1].yaxis.tick_right()

    axes[1, 0].plot(logged_data['ploss'][:MAX_ITER])
    axes[1, 0].grid()
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylabel(r'$\mathcal{L}_{actor}$', fontsize=PLOT_FONT_SIZE)

    axes[1, 1].plot(logged_data['qloss'][:MAX_ITER])
    axes[1, 1].grid()
    # axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel(r'$\mathcal{L}_{critic}$', fontsize=PLOT_FONT_SIZE)
    axes[1, 1].yaxis.set_label_position("right")
    axes[1, 1].yaxis.tick_right()

    gs = axes[2, 0].get_gridspec()
    # remove the underlying axes
    for ax in axes[2, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[2, :])
    axbig.set_ylabel(r'$v_{vio}$', fontsize=PLOT_FONT_SIZE)
    axbig.plot(logged_data['vvio'][:MAX_ITER], color='r', alpha=0.8)
    axbig.grid()
    axbig.set_xlabel('Iteration', fontsize=PLOT_FONT_SIZE)

    for ax in axes:
        for i in range(2):
            ax[i].tick_params(labelsize=PLOT_FONT_SIZE)
    axbig.tick_params(labelsize=PLOT_FONT_SIZE)

    plt.tight_layout()

    plt.savefig('fig2.png', dpi=400)
    plt.show()


if __name__ == '__main__':

    data_dir = 'paper_results/learning_curves'

    name = 'default_experiment'

    plot_learning_curves(data_dir, name)
