# Transformer Spell-Checker

A character-level encoder-decoder Transformer that repairs corrupted English sentences. Its training metrics looked healthy while its real output was worse than doing nothing. Finding and fixing that gap took character error rate (CER) from 0.386 to 0.022 and lifted exact repair of corrupted sentences from 0% to 52%. Built for a Penn State graduate NLP course (Summer 2026).

## The problem

A model can pass every training check and still fail in use. Here the training loop fed the model the correct previous character at every step (teacher forcing), so validation loss looked fine. In real use the model reads its own output, and small mistakes compounded across about 90 steps per sentence. The lesson carries to any generative system: evaluate on the path the model will actually run, against a baseline that shows whether it helps at all.

## Approach

1. **Set an honest baseline.** Only about 2% of characters needed fixing, so copying the input unchanged already scored a CER of 0.028. Any useful model had to fix real errors without damaging the other 98%.
2. **Evaluate the real path.** Every metric runs on free-running output, where the model decodes from its own predictions, scored against that copy baseline. Metrics: CER, exact match, exact match on corrupted sentences only, and edit similarity.
3. **Diagnose.** Early runs showed a CER of 0.386, so the model was making text worse. Two causes: exposure bias from teacher forcing, and a logit-scaling explosion that pushed loss near 180.
4. **Fix.**
   - Tied input and output embeddings with matched initialization (std = d_model^-0.5), which gave the model a prior toward copying and kept output logits at unit scale
   - Warmup plus a cosine learning-rate schedule
   - Longer training with early-stopping patience: 70 epochs overnight on an RTX 3060
5. **Compare decoders.** Greedy, length-constrained greedy and beam search, each against the copy baseline.

## Results

Dev set, free-running decoding. Validation perplexity 1.60.

| System | Exact match | Exact match, corrupted only | CER | Edit similarity |
|---|---|---|---|---|
| Copy input (baseline) | 3.9% | 0.0% | 0.028 | 0.966 |
| Greedy | 51.5% | 50.7% | 0.022 | 0.977 |
| Length-constrained greedy | 51.5% | 50.7% | 0.022 | 0.977 |
| Beam search | 52.4% | 51.8% | 0.022 | 0.977 |

## What I learned

- **Model quality fixed the compounding.** On the weak early models, beam search beat greedy by about 13% CER. On the well-trained model all three decoders tied on CER, so decoding tricks stopped mattering.
- **Initialization is a design choice.** Matched initialization on tied embeddings turned an untrainable model into a trainable one.
- **Know when more training stops paying.** Validation loss flattened near epoch 45. The last 25 epochs bought almost nothing, so more capacity or more data is the next lever.

## Run it

`transformer_spell_checker.ipynb` loads a saved checkpoint from `outputs/assignment2/spell_transformer.pt` by default and skips training. The checkpoint and the course dataset are not in this repo. To retrain, set `USE_EXISTING_CHECKPOINT=False` and `RUN_PROFILE="full"` (about 6 hours on an RTX 3060).
