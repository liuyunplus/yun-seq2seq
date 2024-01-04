import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, enc_input_size, enc_embed_dim, enc_hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(enc_input_size, enc_embed_dim)
        self.lstm = nn.LSTM(enc_embed_dim, enc_hidden_dim, num_layers, dropout=dropout, bidirectional=True)

    def forward(self, input):
        # input shape: (src_len, batch_size)
        embedding = self.dropout(self.embedding(input))
        # embedding shape: (src_len, batch_size, enc_embed_dim)
        output, (hidden, cell) = self.lstm(embedding)
        # output shape: (src_len, batch_size, enc_hidden_dim * 2)
        # hidden shape: (num_layers * 2, batch_size, enc_hidden_dim)
        # cell shape: (num_layers * 2, batch_size, enc_hidden_dim)
        return output, hidden, cell


class Attention(nn.Module):

    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(dec_hidden_dim + enc_hidden_dim, dec_hidden_dim, bias=False)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_output):
        # hidden shape: (num_layers * 2, batch_size, dec_hidden_dim)
        # enc_output shape: (src_len, batch_size, enc_hidden_dim * 2)
        src_len = enc_output.shape[0]
        hidden = hidden.transpose(0, 1)
        # hidden shape: (batch_size, 2 * num_layers, dec_hidden_dim)
        hidden = hidden.reshape(hidden.shape[0], 1, -1)
        # hidden shape: (batch_size, 1, dec_hidden_dim * num_layers * 2)
        hidden = hidden.repeat(1, src_len, 1)
        # hidden shape: (batch_size, src_len, dec_hidden_dim * num_layers * 2)
        enc_output = enc_output.transpose(0, 1)
        # enc_output shape: (batch_size, src_len, enc_hidden_dim * 2)
        mix_input = torch.cat((hidden, enc_output), dim=2)
        # mix_input shape: (batch_size, src_len, dec_hidden_dim * num_layers * 2 + enc_hidden_dim * 2)
        energy = torch.tanh(self.attn(mix_input))
        # energy shape: (batch_size, src_len, dec_hidden_dim)
        attention = self.v(energy).squeeze(2)
        # attention shape: (batch_size, src_len)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):

    def __init__(self, dec_input_size, dec_embed_dim, dec_hidden_dim, output_size, num_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.embedding = nn.Embedding(dec_input_size, dec_embed_dim)
        self.lstm = nn.LSTM(dec_embed_dim + dec_hidden_dim * 2, dec_hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(dec_hidden_dim * 2, output_size)

    def forward(self, input, enc_output, hidden, cell):
        # input shape: (batch_size), but we want (1, batch_size)
        # hidden shape: (enc_num_layers * 2, batch_size, enc_hidden_dim)
        # cell shape: (enc_num_layers * 2, batch_size, enc_hidden_dim)
        input = input.unsqueeze(0)
        embedding = self.embedding(input)
        # embedding shape: (1, batch_size, dec_embed_dim)
        embedding = self.dropout(embedding)

        attn_weights = self.attention(hidden, enc_output)
        # attn_weights shape: (batch_size, src_len)
        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights shape: (batch_size, 1, src_len)
        # enc_output shape: (src_len, batch_size, enc_hidden_dim * 2)
        enc_output = enc_output.transpose(0, 1)
        # enc_output shape: (batch_size, src_len, enc_hidden_dim * 2)
        context = torch.bmm(attn_weights, enc_output)
        # context shape: (batch_size, 1, enc_hidden_dim * 2)
        context = context.transpose(0, 1)
        # context shape: (1, batch_size, enc_hidden_dim * 2)
        rnn_input = torch.cat((embedding, context), dim=2)
        # rnn_input shape: (1, batch_size, enc_hidden_dim * 2 + dec_embed_dim)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # output shape: (1, batch_size, dec_hidden_dim * 2)
        output = self.fc(output)
        # output shape: (1, batch_size, dec_vocab_size)
        output = output.squeeze(0)
        # output shape: (batch_size, dec_vocab_size)
        # hidden shape: (1, batch_size, dec_hidden_dim)
        # cell shape: (1, batch_size, dec_hidden_dim)
        return output, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        max_len = target.shape[0]
        target_vocab_size = self.decoder.output_size
        enc_outputs, hidden, cell = self.encoder(source)
        # enc_outputs shape: (src_len, batch_size, enc_hidden_dim * 2)
        # hidden shape: (enc_num_layers * 2, batch_size, enc_hidden_dim)
        # cell shape: (enc_num_layers * 2, batch_size, enc_hidden_dim)
        outputs = torch.zeros(max_len, batch_size, target_vocab_size)
        # Grab start token
        output = target[0]
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(output, enc_outputs, hidden, cell)
            # output shape: (batch_size, dec_vocab_size)
            # hidden shape: (1, batch_size, dec_hidden_dim)
            # cell shape: (1, batch_size, dec_hidden_dim)
            outputs[t] = output
            top1 = output.argmax(1)
            # top1 shape: (batch_size)
            output = target[t] if random.random() < teacher_force_ratio else top1
        return outputs


def simpleSeq2Seq(source_vocab, target_vocab, embed_dim=32, hidden_dim=64, num_layers=1):
    enc_input_size = len(source_vocab)
    enc_embed_dim = embed_dim
    enc_hidden_dim = hidden_dim
    enc_num_layers = num_layers
    enc_dropout = 0.5 if num_layers > 1 else 0

    dec_input_size = len(target_vocab)
    dec_embed_dim = embed_dim
    dec_hidden_dim = hidden_dim
    dec_output_size = len(target_vocab)
    dec_num_layers = num_layers
    dec_dropout = 0.5 if num_layers > 1 else 0

    attention = Attention(enc_hidden_dim * 2, dec_hidden_dim * dec_num_layers * 2)
    encoder_net = Encoder(enc_input_size, enc_embed_dim, enc_hidden_dim, enc_num_layers, enc_dropout)
    decoder_net = Decoder(dec_input_size, dec_embed_dim, dec_hidden_dim, dec_output_size, dec_num_layers,
                          dec_dropout,
                          attention)
    seq2seq = Seq2Seq(encoder_net, decoder_net)
    return seq2seq