import re
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def sentence_tokenize(txt):
    txt = re.sub(r"[,]", " , ", txt)
    txt = re.sub(r"[.'№:]", " ", txt)
    txt = re.sub(r'["«»]', " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.lower().split()


def tokenize_corpus(texts, tokenizer=sentence_tokenize, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))


def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    """
    :param model: torch.nn.Module - was learned model
    :param dataset: torch.utils.data.Dataset
    :param device: cuda/cpu
    :param batch_size: batch size
    :return: numpy.array размерности len(dataset) x *
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        # import tqdm
        # for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
        for batch_x, batch_y in dataloader:
            batch_x = copy_data_to_device(batch_x, device)

            if return_labels:
                labels.append(batch_y.numpy())
            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)


class StackedConv1d(nn.Module):
    def __init__(self, features_num, layers_n=1, kernel_size=3, conv_layer=nn.Conv1d, dropout=0.0):
        super().__init__()
        layers = []
        for _ in range(layers_n):
            layers.append(nn.Sequential(
                conv_layer(features_num, features_num, kernel_size, padding=kernel_size // 2),
                nn.Dropout(dropout),
                nn.LeakyReLU()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """x - BatchSize x FeaturesNum x SequenceLen"""
        for layer in self.layers:
            x = x + layer(x)
        return x


class SentenceLevelPOSTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size=32, single_backbone_kwargs={},
                 context_backbone_kwargs={}):
        super().__init__()
        self.embedding_size = embedding_size
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
        self.context_backbone = StackedConv1d(embedding_size, **context_backbone_kwargs)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Conv1d(embedding_size, labels_num, 1)
        self.labels_num = labels_num

    def forward(self, tokens):
        """tokens - BatchSize x MaxSentenceLen x MaxTokenLen"""
        batch_size, max_sent_len, max_token_len = tokens.shape
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)

        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen
        char_features = self.single_token_backbone(char_embeddings)

        token_features_flat = self.global_pooling(char_features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize

        token_features = token_features_flat.view(batch_size, max_sent_len,
                                                  self.embedding_size)  # BatchSize x MaxSentenceLen x EmbSize
        token_features = token_features.permute(0, 2, 1)  # BatchSize x EmbSize x MaxSentenceLen
        context_features = self.context_backbone(token_features)  # BatchSize x EmbSize x MaxSentenceLen
        logits = self.out(context_features)  # BatchSize x LabelsNum x MaxSentenceLen
        return logits


class POSTagger:
    def __init__(self, model_path, char_path, label_path, model_size_path):
        with open(char_path, "r") as file:
            self.char2id = json.load(file)
        with open(label_path, "r") as file:
            label2id = json.load(file)
        self.id2label = [tag[0] for tag in sorted(label2id.items(), key=lambda x: x[1])]
        with open(model_size_path, "r") as file:
            model_size = json.load(file)
        self.model = SentenceLevelPOSTagger(len(self.char2id), len(self.id2label),
                                            embedding_size=64,
                                            single_backbone_kwargs=dict(layers_n=3, kernel_size=3, dropout=0.3),
                                            context_backbone_kwargs=dict(layers_n=3, kernel_size=3, dropout=0.3))
        self.model.load_state_dict(
            torch.load(str(model_path), map_location=torch.device('cpu'))
        )
        self.max_sent_len = model_size['max_sent_len']
        self.max_token_len = model_size['max_token_len']

    def __call__(self, sentences):
        sentences = [text.lower() for text in sentences]
        tokenized_corpus = tokenize_corpus(sentences, tokenizer=sentence_tokenize)

        inputs = torch.zeros((len(sentences), self.max_sent_len, self.max_token_len + 2), dtype=torch.long)

        for sent_i, sentence in enumerate(tokenized_corpus):
            tokenized_corpus[sent_i] = sentence[:self.max_sent_len]
            for token_i, token in enumerate(sentence[:self.max_sent_len]):
                sentence[token_i] = token[:self.max_token_len]
                for char_i, char in enumerate(token[:self.max_token_len]):
                    inputs[sent_i, token_i, char_i + 1] = self.char2id.get(char, 0)

        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        predicted_probs = predict_with_model(self.model, dataset)  # SentenceN x TagsN x MaxSentLen
        predicted_classes = predicted_probs.argmax(1)

        result = []
        for sent_i, sent in enumerate(tokenized_corpus):
            result.append([self.id2label[cls] for cls in predicted_classes[sent_i, :len(sent)]])
        return tokenized_corpus, result
