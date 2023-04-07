# Odinsynth Spec to Rule sequence generation

## TODOs for EMNLP:
- Build HF dataset:
  - [x] Synthetic surface rules
  - [ ] Synthetic syntax rules
- [x] Spin up the CodeGen baseline.
- [ ] Test the baseline using our [dataset](https://huggingface.co/datasets/enoriega/odinsynth_sequence_dataset).
- [x] Put together encoder-decoder architecture
- [ ] Write training script
- [ ] Define evaluation
  - [ ] Test rule generation with chat gpt?
- Run experiments:
  - [ ] Fully lexicalized rules (end-to-end rule generation)
  - [ ] Delexicalized rules (generate rule skeletons)
    - Lexicalize rules using:
      - [ ] Language model
      - [ ] Heuristic/Algorithm