from cnn.network import CharacterPredictor
from rnn.rnn_network import build_rnn




class SceneTextReader():

    def __init__(self, cnn_model, rnn_model):
        self.rnn_model = rnn_model

        self.character_predictor = CharacterPredictor(cnn_model)

