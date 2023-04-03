# This is a sample Python script.
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import itertools as it

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict

from BertForSpecificationEncoding import BertForSpecificationEncodingConfig, BertForSpecificationEndocing
from SpecRuleEncoderDecoderModel import SpecRuleEncoderDecoderModel, SpecRuleEncoderDecoderConfig


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

@dataclass
class SpecificationCollator:
    # tokenizer: PreTrainedTokenizerBase
    max_spec_seqs: Optional[int] = None

    start_tag: str = "<sp>"
    end_tag: str = "</sp>"

    def __post_init__(self):
        # Any extra validations after construction will go here
        pass

    def __insert_span_tokens(self, spec: Dict[str, Any]):
        seqs = list()
        pairs = it.islice(zip(spec['words'], spec['match_start'], spec['match_end']), self.max_spec_seqs) \
            if self.max_spec_seqs else zip(spec['words'], spec['match_start'], spec['match_end'])
        for tokens, start, end in pairs:
            tokens = tokens[:start] + [self.start_tag] + tokens[start:end] + [self.end_tag] + tokens[end:]
            seqs.append(' '.join(tokens))

        return seqs

    def __call__(self, example: List[Dict[str, Any]]) -> Dict[str, Any]:
        # We are going to prepend the rule with each of the sentences in the spec
        specification = example["spec"]
        seq = self.__insert_span_tokens(specification)
        specification["annotated_sentences"] = seq

        return example


def tokenize_encoder(datum, tokenizer):
    # Tokenize the spec sentences
    spec_ids = tokenizer(
        datum['spec']['annotated_sentences'], return_tensors='pt', padding=True)

    return spec_ids


def tokenize_decoder(datum, tokenizer):
    # Tokenize the output rule
    rule_ids = tokenizer(datum['rule'], return_tensors='pt')
    # Add the "labels" to the decoder_inputs
    rule_ids['labels'] = torch.cat(
        (rule_ids['input_ids'], torch.Tensor(tokenizer.encode(tokenizer.eos_token)).unsqueeze(0)), dim=-1).long()

    return rule_ids


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = load_dataset("enoriega/odinsynth_sequence_dataset")['train']
    collator = SpecificationCollator(max_spec_seqs=5)
    input_dataset = dataset.map(collator)

    encoder_config = BertForSpecificationEncodingConfig.from_pretrained("bert-base-uncased")
    decoder_config = AutoConfig.from_pretrained("gpt2")

    encoder = BertForSpecificationEndocing.from_pretrained("bert-base-uncased", config=encoder_config)
    decoder = AutoModelForCausalLM.from_pretrained("gpt2", config=decoder_config)

    encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    config = SpecRuleEncoderDecoderConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config
    )

    model = SpecRuleEncoderDecoderModel(encoder, decoder, config)
    encoder_feats = input_dataset.map(
        lambda d: tokenize_encoder(d, encoder_tokenizer),
        remove_columns=input_dataset.column_names
    )
    encoder_feats.set_format("torch")
    decoder_feats = input_dataset.map(
        lambda d: tokenize_decoder(d, decoder_tokenizer),
        remove_columns=input_dataset.column_names
    )
    decoder_feats.set_format("torch")

    e = encoder_feats[0:2]
    d = decoder_feats[0:2]

    x = model(encoder_inputs=e,
              decoder_inputs=d)
    y = 0

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
