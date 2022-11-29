import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Callable
import config
import random

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        rnn_type: Literal["LSTM","RNN","GRU"] = "GRU",
        num_layers: int = 2,
        dropout: float = 0.0,
        ):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        
        rnn_obj = getattr(nn, rnn_type)
        self.rnn = rnn_obj(emb_size, hidden_size, num_layers, bias = False, batch_first = True, dropout = dropout, bidirectional = False)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, inputs):
        ## Input: (bs, enc_seq_length)
        emb = self.dropout(self.emb(inputs)) # (bs, enc_seq_length, enc_vocab_size)
        if self.rnn_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(emb)  
            return outputs, (hidden, cell) # (bs, enc_seq_length, hidden_size), (no_directions * num_layers, bs, hidden_size), (no_directions * num_layers, bs, hidden_size)
        else:
            outputs, hidden = self.rnn(emb)
            return outputs, hidden  # (bs, seq_length, hidden_size), (no_directions * num_layers, bs, hidden_size)

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        rnn_type : Literal["LSTM","RNN","GRU"] = "GRU",
        num_layers: int = 2,
        dropout: float = 0.0,
        attention: bool = True,
    ):
          
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.use_attention = attention
        
        rnn_obj = getattr(nn, rnn_type)
        self.rnn = rnn_obj(emb_size, hidden_size, num_layers, bias = False, batch_first = True)
        self.dropout = nn.Dropout(p = dropout)
        
        if self.use_attention:
            self.output = nn.Linear(hidden_size * 2, vocab_size)
        else:
            self.output = nn.Liner(hidden_size, vocab_size)
            
    def attention(self, encoder_hiddens, decoder_hidden):
        ## Encoder Hiddens: (bs, enc_len, hidden_size)
        ## Decoder hidden: (bs, hidden_size)
        att_weight = torch.bmm(encoder_hiddens, decoder_hidden.unsqueeze(2)) # Att_weight = (bs, enc_len, 1)
        att_weight = F.softmax(att_weight.squeeze(2), dim = 1) # Att_weight = (bs, enc_len)
        
        att_output = torch.bmm(encoder_hiddens.transpose(1,2), att_weight.unsqueeze(2)).squeeze(2) # (bs, hidden_size)
        att_combined = torch.cat((att_output, decoder_hidden), dim = 1) # (bs, hidden_size * 2)
        return att_combined

    def forward(self, inputs, encoder_hiddens, encoder_final_hidden):
        ## Inputs: (bs)
        ## Encoder_hiddens: (bs, enc_seq, hidden_size)
        inputs = inputs.unsqueeze(1) # (bs, 1)
        x = self.dropout(self.emb(inputs)) # (bs, 1, dec_vocab_size)
        
        x, hidden = self.rnn(x, encoder_final_hidden) # (bs, 1, enc_hidden_size), (bs, no_of_directions * num_layers, hidden_size)
        if self.use_attention:
            x = self.attention(encoder_hiddens, x.squeeze(1)) # (bs, hidden_size * 2)
        x = self.output(x) # (bs, dec_vocab_size)
        out = F.log_softmax(x, dim = 1) # (bs, dec_vocab_size)
        return out, hidden # (bs, dec_vocab_size), (bs, no_of_directions * num_layers, hidden_size)

class SeqToSeq(nn.Module):
    def __init__(
        self,
        encoder: Callable,
        decoder: Callable,
    ):
        """Sequence To Sequence Model. 
        Args:
            encoder (Callable): Encoder Unit
            decoder (Callable): Decoder Unit

        Raises:
            Exception: HIdden Size of Encoder must equal Decoder.
        """
        super(SeqToSeq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        if self.encoder.hidden_size != self.decoder.hidden_size:
            raise Exception("Hidden Size mismatch")
        
    def forward(self, inputs: torch.tensor, targets: torch.tensor, teacher_forcing_ratio: float = 0.5):
        """Forward Propagation

        Args:
            inputs (torch.tensor): Inputs sequence to encoder. Dimension: (bs, seq_len)
            targets (torch.tensor): Target sequence to decoder. Dimension: 
            teacher_forcing_ratio (float, optional): _description_. Defaults to 0.0.
        """
        ## Inputs  (bs, enc_seq_len)
        ## Targets (bs, target_seq_len)
        batch_size, target_len = targets.shape
        target_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(config.DEVICE) # (bs, target_len, target_vocab_size)
        encoder_hiddens, encoder_final_hidden = self.encoder(inputs) # (bs, enc_seq_len, hidden_size), (num_layers*num_directions, bs, hidden_size)
        decoder_next_input = targets[:,0] # (bs)
        
        for time in range(1, target_len):
            output, encoder_final_hidden = self.decoder(decoder_next_input, encoder_hiddens, encoder_final_hidden) # (bs, dec_vocab_size), (bs, no_of_directions * num_layers, hidden_size)
            outputs[:,time,:] = output # (bs, dec_vocab_size)
            teacher_force = random.random()
            pred = output.argmax(1) # (bs)
            decoder_next_input = pred if teacher_force > teacher_forcing_ratio else targets[:,time] # (bs)

        return outputs
        
            
if __name__ == "__main__":
    encoder = Encoder(vocab_size = 5000,
                      emb_size = 300,
                      hidden_size = 64,
                      rnn_type = "LSTM")
    
    decoder = Decoder(vocab_size = 3000,
                      emb_size = 300,
                      hidden_size = encoder.hidden_size,
                      rnn_type = encoder.rnn_type,
                      num_layers = encoder.num_layers,
                      attention = True,)
    
    seq2seq = SeqToSeq(encoder, decoder)
    
    sample_inputs = torch.randint(0, 4999, size = (5,20))
    sample_targets = torch.randint(0, 3000, size = (5,15))
    
    out = seq2seq(sample_inputs, sample_targets)
    print(out.size())