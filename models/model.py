# system
import os
import time
import sys
import pickle
from abc import ABC, abstractmethod

# lib
from dataclasses import dataclass
import matplotlib.pyplot as plt

# self
from general.utils import generate_nonce
from general.logger import Logger
# from keras.utils.generic_utils import


@dataclass
class PoliticalSentimentModel(ABC):
    """
    a Model object for training model(s)

    ******** Additional Required Implementation

    the subclass of this object should inherit a __call__ function
    that will perform fitting for that specific model. However, to
    avoid violating LSP, this method is not given with no implementation
    here. This method should, in addition to fitting the agent(s), set
    the attribute of self.history. This will allow one to call
    self.visualize fluidly. The format of self.history should be a
    dictionary keyed with metric names, valued with an additional dictionary
    keyed with agent names and valued with a list of the metric value at every
    epoch. I.E:

        {
            'loss': {
                'agent_1': [2,1,0,...],
                'agent_2': [3,2,1,...],
            },
            'val_loss': {
                'agent_1': [2,1,0,...],
                'agent_2': [nan,nan,3,2,1,...],
            }
        }

    all agents need not be present in every metric, if it is not relevent.
    nans are also allowed if the agent only has the metric on some epochs

    Global model implicit parameters are stated here and come first in
    initialization per PEP 557.

    ******** Paramaters

    :param verbose: the logging level of the of the model
    :param dir: the directory to save models/other data (like
                visualizations). During training, model weights
                are saved in dir/weights/
    """
    verbose: int = 1
    dir: str = './bin/' + time.strftime("%Y-%m-%d_%H:%M") + "__" + generate_nonce()

    @abstractmethod
    def initialize_model(self):
        """
        initialize the model and compile it.
        :return: a list of the model agent(s) involved
        """
        raise NotImplementedError()

    def __post_init__(self):
        """
        after data initialization runs from the dataclass, Also, create
        the model directory if it is not already present.

        initialize the model and also compile it. Do any instance
        specific setup here as well.
        """

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.logger = Logger(os.path.join(self.dir, 'log.txt'))
        sys.stdout = self.logger
        self.agents = self.initialize_model()

    def _get_save_path(self, agent_index):
        """
        get the save path for the agent with a given index
        :param agent_index: the index of the agent.
        :return: the save path
        """
        return os.path.join(self.dir, 'weights/agent_' + str(agent_index))

    def save(self):
        """
        save the state of the models in their save directory
        :return: None
        """
        weights_dir = os.path.join(self.dir, 'weights/')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        for i, agent in enumerate(self.agents):
            agent.save_weights(self._get_save_path(i))

        if hasattr(self, 'word_to_index'):
            meta_dir = os.path.join(self.dir, 'meta/')
            if not os.path.exists(meta_dir):
                os.makedirs(meta_dir)

            with open(os.path.join(meta_dir, 'word_to_index.pkl'), 'wb') as f:
                pickle.dump(self.word_to_index, f)

    def load(self):
        """
        load the state of thhe model from a saved state.
        :return: None
        """
        print('loading')
        for i, agent in enumerate(self.agents):
            agent.load_weights(self._get_save_path(i))

        meta_dir = os.path.join(self.dir, 'meta/')
        if os.path.exists(os.path.join(meta_dir, 'word_to_index.pkl')):
            with open(os.path.join(meta_dir, 'word_to_index.pkl'), 'rb') as f:
                self.word_to_index = pickle.load(f)

    def print(self, *args):
        """
        print based on the model verbosity
        :param args: what to print.
        :return: None
        """
        if self.verbose:
            print(*args)

    def visualize(self, metrics_to_mix=(), sub_directory=None):
        """
        visualize the history of a model's performance among the
        various metrics it was compiled with with all of its agents.

        the plots are saved in self.dir/visualize/...

        :param metrics_to_mix: a list (or tuple) of sets of names of metrics
                               to put on the same graph. The plot title and y
                               axis will use the first name in this set, with any
                               strings 'val_' removed. For example,

                               metrics_to_mix = [set(['val_acc', 'acc'])]

                               will have validation accuracy and accuracy on the same plot
                               that will have acc used as the naming convention.

        :param sub_directory: a sub-directory to put the visualization in. This is useful
                              when you are experimenting by training the same network many times.

        :return: None.
        """

        if not hasattr(self, 'history'):
            raise ValueError('the model must be fit first')

        sub_directory = sub_directory if sub_directory is not None else ''
        visualization_dir = os.path.join(os.path.join(self.dir, sub_directory), 'visualization/')
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)

        unspecified = set(self.history.keys()) - set().union([e for i in metrics_to_mix for e in i])
        metrics = list(metrics_to_mix) + [[e] for e in unspecified]
        print(metrics)

        for metric_set in metrics:
            if len(metric_set) == 0:
                raise ValueError('empty metric set given. do not do that.')

            legend = []
            for metric in metric_set:
                agents = self.history[metric]
                for agent, scores in agents.items():
                    plt.plot(scores)
                    legend.append(agent + ' ' + metric)

                metric_set_name = metric_set[0]
                metric_set_name.replace('val_', '')

            plt.title('model ' + metric_set_name)
            plt.ylabel(metric_set_name)
            plt.xlabel('epoch')
            plt.legend(legend, loc='upper right')
            plt.savefig(os.path.join(visualization_dir, ','.join(metric_set) + '.png'))
            plt.clf()

    @staticmethod
    def generate_cohesive_history(histories):
        """
        generate a cohesive history object like the one expected
        by visualize.

        :param histories: a dictionary keyed with agents and valued
                          with their (joined) keras history.history objects.
        :return: the history object formatted described above.
        """

        metric_names = list(set([metric_name for history in histories.values() for metric_name in history.keys()]))

        history = {
            metric_name: {agent: value[metric_name] for agent, value in histories.items() if metric_name in value}
            for metric_name in metric_names
        }

        return history


