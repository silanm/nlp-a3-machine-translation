import random

import torch
import torch.nn as nn
import torchdata.datapipes as dp
from datasets import load_dataset
from pythainlp.tokenize import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def restructure_dataset(dataset):
    return [(item["translation"]["en"], item["translation"]["th"]) for item in dataset]


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def yield_tokens(data, language):
    language_index = {SRC_LANGUAGE: 0, TRG_LANGUAGE: 1}

    for data_sample in data:
        yield token_transform[language](data_sample[language_index[language]])  # either first or second index


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))


class DatasetIterable:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            yield item


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attention_type, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attention_type, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attention_type, device, max_length=640):
        assert attention_type in ["gen", "mult", "add"], "Attention type must be either 'gen', 'mult' or 'add'."

        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.max_length = max_length  # Store max length
        self.pos_embedding = nn.Embedding(max_length, hid_dim)  # # Allow extra room
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, attention_type, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, min(src_len, self.max_length)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = torch.clamp(src, min=0, max=self.tok_embedding.num_embeddings - 1)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, attention_type, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.attention_type = attention_type
        if attention_type == "mult":
            self.W = nn.Parameter(torch.randn(self.head_dim, self.head_dim))  # Learnable weight matrix W
        if attention_type == "add":
            self.W1 = nn.Linear(self.head_dim, self.head_dim)
            self.W2 = nn.Linear(self.head_dim, self.head_dim)
            self.v = nn.Linear(self.head_dim, 1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.attention_type == "gen":
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        elif self.attention_type == "mult":
            energy = torch.matmul(torch.matmul(Q, self.W), K.permute(0, 1, 3, 2)) / self.scale
        elif self.attention_type == "add":
            W1_K = self.W1(K)
            W2_Q = self.W2(Q)
            W2_Q_expanded = W2_Q.unsqueeze(3).expand(-1, -1, -1, K.shape[2], -1)
            energy = self.v(torch.tanh(W1_K.unsqueeze(2) + W2_Q_expanded)).squeeze(-1) / self.scale
            if V.shape[2] != energy.shape[-1]:
                V = V[:, :, : energy.shape[-1], :]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attention_type, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attention_type, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attention_type, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attention_type, device, max_length=640):
        assert attention_type in ["gen", "mult", "add"], "Attention type must be either 'gen', 'mult' or 'add'."

        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.max_length = max_length  # Store max length
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, attention_type, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, min(trg_len, self.max_length)).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = torch.clamp(trg, min=0, max=self.tok_embedding.num_embeddings - 1)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention


SEED = 1527
dataset_fraction = 0.5
batch_size = 16
SRC_LANGUAGE = "en"
TRG_LANGUAGE = "th"
genatt_save_path = "models/20250202002832_Seq2SeqTransformer_genatt_seed1527_nb6261_bs16.pt"
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<sos>", "<eos>"]
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1
SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX
token_transform = {}
vocab_transform = {}
text_transform = {}


def prepare_model() -> Seq2SeqTransformer:
    print("Preparing model...")
    dataset = load_dataset("airesearch/scb_mt_enth_2020", "enth")
    train = dataset["train"].shuffle(seed=SEED).select(range(int(dataset_fraction * len(dataset["train"]))))
    test = dataset["test"].shuffle(seed=SEED).select(range(int(dataset_fraction * len(dataset["test"]))))
    train_datapipe = dp.iter.IterableWrapper(DatasetIterable(train))
    test_datapipe = dp.iter.IterableWrapper(DatasetIterable(test))
    sharded_train_datapipe = dp.iter.ShardingFilter(train_datapipe)
    sharded_test_datapipe = dp.iter.ShardingFilter(test_datapipe)
    sharded_train_datapipe.apply_sharding(num_of_instances=4, instance_id=0)
    sharded_test_datapipe.apply_sharding(num_of_instances=4, instance_id=0)
    train = sharded_train_datapipe
    test = sharded_test_datapipe
    train = restructure_dataset(train)
    test = restructure_dataset(test)
    token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")
    token_transform[TRG_LANGUAGE] = lambda x: word_tokenize(x, engine="newmm")

    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        vocab_transform[ln] = build_vocab_from_iterator(
            yield_tokens(train, ln),
            min_freq=2,
            specials=special_symbols,
            special_first=True,
        )

    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],
            vocab_transform[ln],
            tensor_transform,
        )
    input_dim = len(vocab_transform[SRC_LANGUAGE])
    output_dim = len(vocab_transform[TRG_LANGUAGE])
    enc_gen = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, attention_type="gen", device="cpu")
    dec_gen = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, enc_dropout, attention_type="gen", device="cpu")
    model_gen_test = Seq2SeqTransformer(enc_gen, dec_gen, SRC_PAD_IDX, TRG_PAD_IDX, device="cpu")
    model_gen_test.load_state_dict(torch.load(genatt_save_path))
    model_gen_test.eval()
    print("Model prepared.")
    return test, model_gen_test


def randomize_text(test):
    print("Randomizing text...")
    random_text = random.choice(test)
    print("Text randomized: ", random_text)
    return random_text


def translate_text(model_gen_test, random_text) -> str:
    print("Translating text...")
    src_text = text_transform[SRC_LANGUAGE](random_text[0])
    src_text = src_text.reshape(1, -1)
    trg_text = text_transform[TRG_LANGUAGE](random_text[1])
    trg_text = trg_text.reshape(1, -1)
    # text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)
    with torch.no_grad():
        output_genatt, attentions_genatt = model_gen_test(src_text, trg_text)
    output_genatt = output_genatt.squeeze(0)
    output_genatt = output_genatt[1:]
    output_genatt_max = output_genatt.argmax(1)  # returns max indices
    mapping = vocab_transform[TRG_LANGUAGE].get_itos()
    translated_sentence_genatt = "".join([mapping[token.item()] for token in output_genatt_max if mapping[token.item()] != "<eos>"])
    print("Text translated: ", translated_sentence_genatt)
    return translated_sentence_genatt


# test, model_gen_test = prepare_model()
# sample = randomize_text(test)
# translated_text = translate_text(sample)
