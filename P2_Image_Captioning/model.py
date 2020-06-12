import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Initialize the model by setting up the layers.
        """
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weigths()
    
    def forward(self, features, captions):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        lengths = captions.shape[1]
        embeds = self.embedding(captions[:, :-1])
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        hiddens, _ = self.lstm(embeds)

        # Final projection
        outputs = self.fc(hiddens)

        return outputs
        

    def init_weigths(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption_pred = []
        count = 0
        word_token = None
        
        while count < max_len and word_token != 1:
            output_lstm, states = self.lstm(inputs, states)
            output = self.fc(output_lstm)
            
            prob, word = output.max(2)
            word_token = word.item()
            caption_pred.append(word_token)
            
            inputs = self.embedding(word)
            
            count +=1 
        return caption_pred
