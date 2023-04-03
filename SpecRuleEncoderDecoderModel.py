from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, GPT2Config, AutoModel, AutoModelForCausalLM, GPT2LMHeadModel
from transformers.utils import ModelOutput

import BertForSpecificationEncoding
from BertForSpecificationEncoding import BertForSpecificationEncodingConfig, SpecificationEncodingOutput


class SpecRuleEncoderDecoderConfig(PretrainedConfig):
    encoder_config: BertForSpecificationEncodingConfig
    decoder_config: GPT2Config
    is_composition = True  # TODO: Figure out how to mix two model configurations


@dataclass
class SpecRuleEncoderDecoderOutput(ModelOutput):
    # TODO fill here
    ...


class SpecRuleEncoderDecoderModel(PreTrainedModel):
    def __init__(self, encoder: BertForSpecificationEncoding, decoder: GPT2LMHeadModel, config: SpecRuleEncoderDecoderConfig):
        super().__init__(config)
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        # self.encoder = BertForSpecificationEncoding.from_pretrained(config.encoder_config)
        # self.decoder = AutoModelForCausalLM.from_config(config.decoder_config)

    def forward(
            self,
            encoder_inputs,
            decoder_inputs: Optional[dict] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], SpecRuleEncoderDecoderOutput]:

        # Encoder pass
        encoder_output = self.encoder(
            **encoder_inputs,
        )

        # Collect the encoded specifications to be used as seeds in the decoding step
        seed_embeddings = encoder_output.embeds.unsqueeze(dim=0)

        # We will insert the seed embeddings as the first input to the CausalLM
        # For this, we will explicitly pass the word embeddings as input to the decoder
        if decoder_inputs:
            # Insert the seed embeddings in the second dimension (first is the batch dimension)
            decoder_input_embeds = self.decoder.transformer.wte(decoder_inputs['input_ids'])
            decoder_input_embeds = torch.cat((seed_embeddings, decoder_input_embeds), dim=1)
            labels = decoder_inputs['labels']
        else:
            decoder_input_embeds = seed_embeddings
            labels = None

        decoder_output = self.decoder(
            inputs_embeds=decoder_input_embeds,
            labels=labels,
        )

        # Return the CausalLM loss, for now
        return decoder_output
