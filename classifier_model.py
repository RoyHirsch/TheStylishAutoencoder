import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_model import MultiHeadedAttention, PositionwiseFeedForward, BasicEncoder, EncoderLayer
import copy


class MaskedMean(nn.Module):
    " Calculate masked mean of input 3D tensor "

    def __init__(self, normalize=True):
        self.normalize = normalize
        super().__init__()

    def forward(self, x, mask):
        mask = mask.squeeze(1)
        batch_size, _, embed_size = x.size()
        mask_expend = torch.stack(embed_size * [mask], 2).float()
        masked_input = x * mask_expend
        sum_ = torch.sum(masked_input, 1)
        div = mask.sum(1).float()
        embed = torch.div(sum_, div.view(batch_size, 1))
        if self.normalize:
            return F.normalize(embed, p=2, dim=1)
        else:
            return embed


class TransformerClassifier(nn.Module):
    """
    Transformer for style classification
    """

    def __init__(self, output_size, input_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = BasicEncoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.generator = nn.Linear(d_model, output_size)
        self.masked_mean = MaskedMean(False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        src = self.embed(src)
        out = self.encoder(src, src_mask)
        out = self.generator(out)
        out = self.masked_mean(out, src_mask)
        return out


class DescriminatorAttention(nn.Module):
    ''' https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch '''

    def __init__(self, output_size, hidden_size,
                 embedding_length, drop_rate):
        super().__init__()

        self.hidden_size = hidden_size
        # Encoder RNN
        self.lstm = nn.LSTM(input_size=embedding_length,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=True)

        # Dropout Layer
        self.dropout = nn.Dropout(drop_rate)

        # Fully-Connected Layer
        self.fc = nn.Linear(hidden_size * 4, output_size)

    def apply_attention(self, rnn_output, final_hidden_state):
        '''
        Apply Attention on RNN output

        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN

        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        '''
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output

    def forward(self, x):
        x = x.permute(1, 0, 2)
        # x.shape = (max_sen_len, batch_size, embed_size)

        ##################################### Encoder #######################################
        lstm_output, (h_n, c_n) = self.lstm(x)
        # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)

        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(1, 2, batch_size, self.hidden_size)[-1, :, :, :]

        ##################################### Attention #####################################
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i, :, :] for i in range(h_n_final_layer.shape[0])], dim=1)

        attention_out = self.apply_attention(lstm_output.permute(1, 0, 2), final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)

        #################################### Linear #########################################
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        final_feature_map = self.dropout(concatenated_vector)  # shape=(batch_size, num_directions * hidden_size)
        final_out = self.fc(final_feature_map)
        return final_out


class Descriminator(nn.Module):
    ''' https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch '''

    def __init__(self, input_size, embedding_size, hidden_size, output_size, drop_rate, num_layers=1):

        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers

        self.embed = nn.Linear(input_size, embedding_size)
        # Encoder RNN
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True)

        # Dropout Layer
        self.dropout = nn.Dropout(drop_rate)

        # Fully-Connected Layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embed(x)

        x = x.permute(1, 0, 2)
        # x.shape = (max_sen_len, batch_size, embed_size)

        lstm_output, (h_n, c_n) = self.lstm(x)
        # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)

        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        batch_size = h_n.shape[1]
        final_hidden = h_n.view(2 * self.n_layers, batch_size, self.hidden_size)
        cat_final_hidden = torch.cat((final_hidden[-2, :, :], final_hidden[-1, :, :]), 1)

        out = self.fc1(cat_final_hidden)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
