import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, modeling_outputs

class FakeEncoder(nn.Module):
    def __init__(self, d_in, S=8, d_model=768):
        super().__init__()
        self.S = S
        self.proj = nn.Linear(d_in, S * d_model)

    def forward(self, x):
        # x: (B, D)
        x = self.proj(x)          # (B, S*768)
        x = x.view(x.size(0), self.S, 768)  # (B, S, 768)

        return x


class DecoderModel(nn.Module):
    def __init__(self, D=768, S=16, d_model=768):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

        self.D = D
        self.S = S
        self.d_model = 768
        self.fake_encoder = FakeEncoder(
            d_in=self.D,
            S=self.S,        
            d_model=self.d_model
        )
    
    def forward(self, gnc_embeddings_outputs):
        B, D = gnc_embeddings_outputs.shape

        fake_hidden_states = self.fake_encoder(gnc_embeddings_outputs)

        encoder_outputs = modeling_outputs.BaseModelOutput(
            last_hidden_state=fake_hidden_states
        )

        ecoder_input_ids = torch.full(
            (B, 1),
            self.model.config.decoder_start_token_id
        )

        embeddings = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=ecoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        tokens = self.model.generate(
                input_ids=None,  # important
                encoder_outputs=encoder_outputs,
                max_new_tokens=100,
            )

        decoder_hidden_states = embeddings.decoder_hidden_states
        texts = self.tokenizer.decode(tokens, skip_special_tokens=True)

        return {"hidden_states": decoder_hidden_states, "texts": texts}




