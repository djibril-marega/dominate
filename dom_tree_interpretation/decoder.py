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


def decoder_t5_pipeline(decoder_inputs_file_path, decoder_outputs_file_path):
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    gnc_embeddings_outputs = torch.load(decoder_inputs_file_path)
    B, D = gnc_embeddings_outputs.shape

    fake_encoder = FakeEncoder(
        d_in=D,
        S=16,        
        d_model=768
    )

    fake_hidden_states = fake_encoder(gnc_embeddings_outputs)

    encoder_outputs = modeling_outputs.BaseModelOutput(
        last_hidden_state=fake_hidden_states
    )
    outputs = model.generate(
        input_ids=None,  # important
        encoder_outputs=encoder_outputs,
        max_new_tokens=100,
    )
    #print(output)
    print(tokenizer.decode(outputs, skip_special_tokens=True))
    torch.save(outputs, decoder_outputs_file_path)




