# coding: utf-8

from keras.engine import Model


class DIINModel(Model):
    def __init__(self, inputs=None, outputs=None, name="DIIN"):
        """Densely Interactive Inference Network(DIIN)

        Model from paper `Natural Language Inference over Interaction Space`
        (https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-)

        :param inputs: inputs of keras models
        :param outputs: outputs of keras models
        :param name: models name
        """

        if inputs or outputs:
            super(DIINModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
            return


