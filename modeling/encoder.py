from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import MSELoss
from transformers import BertConfig, BertPreTrainedModel, BertModel, PretrainedConfig
from transformers.utils import ModelOutput


@dataclass
class SpecificationEncodingOutput(ModelOutput):
    embeds: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    spec_sizes: torch.IntTensor = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertForSpecificationEncodingConfig(BertConfig):
    model_type = "BertForSpecificationEncoding"
    def __init__(
            self,
            rule_sentence_encoding: str = "cls",
            spec_encoding: str = "attention",
            loss_func: str = "mse",
            spec_dropout: float = 0.1,
            margin: Optional[float] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.rule_sentence_encoding = rule_sentence_encoding.lower()
        self.spec_encoding = spec_encoding.lower() if spec_encoding else None
        self.loss_func = loss_func.lower()
        self.spec_dropout = spec_dropout
        self.margin = margin
        self.problem_type = "regression"  # Fixed to regression, because we predict a score for each rule - spec pair


class BertForSpecificationEncoding(BertPreTrainedModel):
    config_class = BertForSpecificationEncodingConfig

    def __init__(self, config: BertForSpecificationEncodingConfig, **kwargs):
        super().__init__(config, kwargs)

        # This will be Robert's ckpt or another BERT/Transformer model
        self.bert = BertModel(config)

        # These are the head's parameters
        spec_dropout = (
            config.spec_dropout if config.spec_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(spec_dropout)
        # This is the regression layer, to predict the encoded spec's score
        self.regressor = nn.Linear(config.hidden_size, 1)

        # If the config requests an attention spec aggregation
        if self.config.spec_encoding == "attention":
            # Linear maps to project the spec into Key and Value spaces
            self.spec_key = nn.Linear(config.hidden_size, config.hidden_size)
            self.spec_value = nn.Linear(config.hidden_size, config.hidden_size)
            # The query will be a single, tunable parameter
            # Dim 1 is of size 1, because we want to aggregate the encoded spec elements into a single pooled representation
            query_store = torch.empty(1, config.hidden_size)
            # Will do xavier initialization on the query parameter
            nn.init.xavier_uniform_(query_store)
            # Make it a model parameter to accept updates
            self.spec_query = nn.Parameter(data=query_store)
            # Instantiate a multi-head attention layer, with a single head
            self.spec_attention = nn.MultiheadAttention(config.hidden_size, 1, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            spec_sizes: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpecificationEncodingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Flatten the inputs if necessary
        # Flatten the inputs for the encoder
        if len(input_ids.shape) == 3:
            spec_sizes = torch.ones((input_ids.size()[0],)).int() * input_ids.size()[1]
            input_ids = input_ids.reshape((-1, input_ids.size()[-1]))
            attention_mask = attention_mask.reshape((-1, attention_mask.size()[-1]))
            token_type_ids = token_type_ids.reshape((-1, token_type_ids.size()[-1]))
            # encoder_inputs = {
            #     k: v.reshape((-1, v.size()[-1])) for k, v in encoder_inputs.items()
            # }
            spec_sizes = spec_sizes.tolist()


        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Encode each rule-sentence pair
        # Assume each tensor in the batch dimension is a rule-sent pair
        # Spec elements are identified by their spec_id value at the same index of the batch dimension

        rule_sent_encoding = self.config.rule_sentence_encoding
        # Use the [CLS] token to represent each rule-sent pair in the spec
        if rule_sent_encoding == "cls":
            pair_embeds = outputs[1]
        # Do pooling
        elif rule_sent_encoding in {"avg", "max"}:
            last_hidden_states = outputs[0]
            # Use the attention mask to zero out the paddigns

            if rule_sent_encoding == "avg":
                last_hidden_states[attention_mask == 0, :] = 0.
                pair_embeds = torch.div(last_hidden_states.sum(dim=1), attention_mask.sum(dim=1).unsqueeze(-1))
            elif rule_sent_encoding == "max":
                # Had to do this clone operation to avoid breaking autograd
                x = last_hidden_states.clone()
                x[attention_mask == 0, :] = last_hidden_states.min()
                pair_embeds = torch.max(x, dim=1)[0]
            else:
                raise ValueError(f"{rule_sent_encoding} is not a valid rule_sentence_encoding option")
        else:
            raise ValueError(f"{rule_sent_encoding} is not a valid rule_sentence_encoding option")

        spec_encoding = self.config.spec_encoding

        if spec_sizes is not None:
            assert sum(spec_sizes) == pair_embeds.size()[0], "Spec sizes must add up to the number of inputs"
        else:
            spec_sizes = [pair_embeds.size()[0]]

        splits = pair_embeds.split(spec_sizes, dim=0)

        embeds = list()
        for split in splits:

            if spec_encoding == "avg":
                spec = split.mean(dim=0)
            elif spec_encoding == "max":
                spec = split.max(dim=0)[0]
            elif spec_encoding == "attention":
                spec = self.spec_attention(
                    self.spec_query,
                    self.spec_key(split),
                    self.spec_value(split)
                )[0].squeeze()
            elif spec_encoding is None:  # No aggregation, i.e. for use with cross attention
                spec = split
            else:
                raise ValueError(f"{spec_encoding} is not a valid spec encoding option")

            embeds.append(spec)

        # Stack the list of tensors into a single tensor with specs
        tensor_embeds = torch.stack(embeds, dim=0)

        loss = None
        # if labels is not None:
        #     labels = labels.squeeze()
        #     if self.config.loss_func == "mse":
        #         loss_fct = MSELoss()
        #         loss = loss_fct(scores, labels)
        #     else:
        #         raise ValueError(f'Loss function must be either "mse" or "margin" and have a default margin value')

        if not return_dict:
            output = (tensor_embeds, torch.tensor(spec_sizes),) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpecificationEncodingOutput(
            loss=loss,
            embeds=tensor_embeds,
            spec_sizes=torch.tensor(spec_sizes, device=self.device),
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
