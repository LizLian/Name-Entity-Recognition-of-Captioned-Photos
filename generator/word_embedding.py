#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Discourse Relation Sense Classifier

Feel free to change/restructure the code below
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
import numpy as np
import json
import spacy


if __name__ == "__main__":
    train_file = "../annotated_data/train.jsonl"
    test_file = "../annotated_data/test.jsonl"
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
    vocab = {"": 0}
    with open(train_file) as infile:
        lines = infile.readlines()

    for line in lines:
        json_line = json.loads(line)
        text = json_line["text"]
        doc = NLP(text)
        for word in doc:
            if word.norm_ not in vocab:
                vocab[word.norm_] = len(vocab)

    # with open(test_file) as testfile:
    #     lines = testfile.readlines()
    # for line in lines:
    #     json_line = json.loads(line)
    #     text = json_line["text"]
    #     doc = NLP(text)
    #     for word in doc:
    #         vocab[word] = len(vocab)

    encoded_docs = []
    max_len = 50
    for line in lines:
        json_line = json.loads(line)
        text = json_line["text"]
        doc = NLP(text)
        encoded_doc = []
        for word in doc:
            encoded_doc.append(vocab[word.norm_])
        encoded_doc = encoded_doc[:max_len]
        encoded_doc += [0] * (max_len - len(encoded_doc)) # padding & length 30
        encoded_docs.append(encoded_doc)

    model = Sequential()
    model.add(Embedding(len(vocab), 100, input_length=max_len))
    model.compile('rmsprop', 'mse')
    input_array = np.asarray(encoded_docs)
    output_array = model.predict(input_array)
    embeddings = {}
    revr_vocab = {vocab[word]: word for word in vocab}
    words = []
    with open("embedding.txt", "w") as outfile:
        for doc_ind in range(len(encoded_docs)):
            doc = encoded_docs[doc_ind]
            for index in range(len(doc)):
                word = revr_vocab[doc[index]]
                if word not in words:
                    outfile.write(word+"\t")
                    output_array[doc_ind][index].tofile(outfile, sep=",")
                    outfile.write("\n")
                    words.append(word)
