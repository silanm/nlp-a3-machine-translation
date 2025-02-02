# nlp-a3-machine-translation

![](images/translation-in-action.gif)

![](images/all_losses_20250202002832_seed1527_nb6261_bs16.svg)

![](images/attention_maps_20250202002832_seed1527_nb6261_bs16.svg)

Use sbc dataset
* sample size: train = XXX, val = XXX, test = XXX
* need to restructure your dataset by transforming each nested dictionary entry into a tuple 
* use PyThaiNLP for tokenizer for Thai languange

To fix index out of range in self
* Check if the vocabulary includes <unk>, <pad>, <sos>, <eos>
* Ensure vocab.set_default_index(UNK_IDX) is set correctly
* Ensure tokens are correctly transformed into indices, If an unknown word appears, it should be mapped to <unk> instead of an out-of-range index.
* Ensure the sequence length doesn’t exceed the predefined max_length:
* since, max_len = 100 hardcoded, check actual max length = 534 -> increase frm 100 to 640
* lastly, print values for debugging

Compare Training time CPU vs. MPS