from typing import Mapping, Sequence, Dict, Optional, Iterable, List

from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from collections import defaultdict, Counter
import json, random, pycrfsuite, spacy, sys, csv, copy
from utils import FeatureExtractor, EntityEncoder, PRF1, PUNC_REPEAT_RE, DIGIT_RE, UPPERCASE_RE, LOWERCASE_RE, ScoringCounts, ScoringEntity
from pymagnitude import Magnitude
from decimal import ROUND_HALF_UP, Context

NLP = spacy.load("en_core_web_sm", disable=["ner"])
data = "train"

def ingest_json_document(doc_json: Mapping, nlp: Language) -> tuple:
    text = doc_json["text"]
    if "review_id" in doc_json:
        id = doc_json["review_id"]
    if "id" in doc_json:
        id = doc_json["id"]
    doc = nlp(text)
    if "annotation_approver" in doc_json:
        annotation_approver = doc_json["annotation_approver"]
    else:
        annotation_approver = ""
    if "labels" in doc_json:
        labels = doc_json["labels"]
    else:
        labels = []
    # raise a value error if annotation approver is none and no annotated labels
    if len(labels) == 0 and annotation_approver is None:
        # raise ValueError("the doc is unannotated")
        print(f"{id} : the doc is unannotated")
        return None
    else:
        try:
            char_offsets = []
            char_start = 0
            for position in range(len(doc)):
                char_start = doc[position].idx
                char_end = char_start + len(doc[position])
                char_offsets.append((char_start, char_end))

            entities = [] # a list of Span objects
            for label in labels:
                start_token = get_token_offset(label[0], char_offsets)
                end_token = get_token_offset(label[1], char_offsets)
                entity_type = label[2]
                if start_token is None or end_token is None:
                    # raise ValueError("start or end token cannot be found")
                    print(f"{id} : start or end token cannot be found")
                    return None
                end_token += 1
                entities.append(Span(doc, start_token, end_token, entity_type))
            doc.ents = entities[:]
        except Exception as e:
            print(f"{id} : {e}")
            return None
    return doc


def get_token_offset(position: int, char_offsets: list):
    for i in range(len(char_offsets)):
        if position >= char_offsets[i][0] and position <= char_offsets[i][1]:
            return i
    return None


def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    isMapped = type_map is not None and len(type_map) > 0
    result = {}
    all_true_pos = 0.0
    all_false_pos = 0.0
    all_false_neg = 0.0
    test_dict = defaultdict(int)
    ref_dict = defaultdict(int)
    for test_doc in test_docs:
        test_ents = test_doc.ents
        for test_ent in test_ents:
            test_dict[test_ent.label_] += 1
    for ref_doc in reference_docs:
        ref_ents = ref_doc.ents
        for ref_ent in ref_ents:
            ref_dict[ref_ent.label_] += 1

    count = defaultdict(defaultdict)
    # label is after mapping
    labels = set(list(ref_dict.keys()) + list(test_dict.keys()))
    for label in labels:
        if isMapped and label in type_map:
            if label in test_dict:
                if type_map[label] not in test_dict:
                    test_dict[type_map[label]] = test_dict[label]
                else:
                    test_dict[type_map[label]] += test_dict[label]
            if label in ref_dict:
                if type_map[label] not in ref_dict:
                    ref_dict[type_map[label]] = ref_dict[label]
                else:
                    ref_dict[type_map[label]] += ref_dict[label]
    if isMapped:
        labels = set(list(ref_dict.keys()) + list(test_dict.keys()))
        for key in type_map:
            labels.remove(key)

    for label in labels:
        true_pos = 0.0
        for i in range(len(test_docs)):
            test_ents = test_docs[i].ents
            ref_ents = reference_docs[i].ents
            for test_ent in test_ents:
                for ref_ent in ref_ents:
                    if test_ent.start == ref_ent.start and test_ent.end == ref_ent.end and \
                            test_ent.label_ == ref_ent.label_ and test_ent.label_ == label:
                        true_pos += 1.0
                    elif isMapped and test_ent.start == ref_ent.start and test_ent.end == ref_ent.end:
                        if test_ent.label_ in type_map and ref_ent.label_ in type_map and \
                                label in type_map.values() and \
                                type_map[test_ent.label_] == type_map[ref_ent.label_] and \
                                type_map[ref_ent.label_] == label:
                            true_pos += 1.0

        false_pos = test_dict[label] - true_pos
        false_neg = ref_dict[label] - true_pos
        if isMapped and label in type_map.values():
            if label not in count:
                count[label]["FP"] = false_pos
                count[label]["FN"] = false_neg
                count[label]["TP"] = true_pos
            else:
                count[label]["FP"] += false_pos
                count[label]["FN"] += false_neg
                count[label]["TP"] += true_pos
        else:
            count[label]["FP"] = false_pos
            count[label]["FN"] = false_neg
            count[label]["TP"] = true_pos

        all_true_pos += true_pos
        all_false_neg += false_neg
        all_false_pos += false_pos

    for label in count:
        prf1 = get_metrics(count[label]["TP"], count[label]["FP"], count[label]["FN"])
        result[label] = prf1
    prf1 = get_metrics(all_true_pos, all_false_pos, all_false_neg)
    result[""] = prf1
    return result


def get_metrics(true_pos, false_pos, false_neg):
    if true_pos + false_pos == 0.0:
        precision = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos + false_neg == 0.0:
        recall = 0.0
    else:
        recall = true_pos / (true_pos + false_neg)
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    prf1 = PRF1(precision, recall, f1)
    return prf1


class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        self.magnitude_word_vector = Magnitude(vectors_path, normalized=False)
        self.scale = scaling

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            word_vector = self.magnitude_word_vector.query(token)
            scaled_word_vector = word_vector * self.scale
            keys = []
            for i in range(len(scaled_word_vector)):
                keys.append("v"+str(i))
            features.update(zip(keys, scaled_word_vector))
        return features


class BrownClusterFeature(FeatureExtractor):
    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ):
        self.use_full_paths = use_full_paths
        self.use_prefixes = use_prefixes
        self.prefixes = prefixes
        self.clusters_path = clusters_path
        self.clusters = {}
        if self.use_full_paths is False and self.use_prefixes is False:
            raise ValueError("use_full_paths or use_prefixes has to be true")
        with open(self.clusters_path) as infile:
            lines = [line for line in infile.read().split("\n") if line]
        for line in lines:
            cluster, word, word_freq = line.split("\t")
            self.clusters[word] = cluster

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0 and token in self.clusters:
            if self.use_full_paths:
                features["cpath="+self.clusters[token]] = 1.0
            if self.use_prefixes:
                if self.prefixes:
                    for prefix in self.prefixes:
                        # only generate for prefixes that are not longer than the path
                        if prefix <= len(self.clusters[token]) and prefix > 0:
                            features["cprefix"+str(prefix)+"="+self.clusters[token][0:prefix]] = 1.0
                else:
                    for i in range(len(self.clusters[token])):
                        features["cprefix"+str(i+1)+"="+self.clusters[token][0:i+1]] = 1.0
        return features


class BiasFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["bias"] = 1.0


class POSFeature(FeatureExtractor):
    def extract(
        self,
        token: Token,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[Token],
        features: Dict[str, float],
    ) -> None:
        if token.pos_ == "ADJ":
            features["pos-adj"] = 1.0
        if token.pos_ == "NOUN":
            features["pos-n"] = 1.0
        # if token.pos_ == "VERB":
        #     features["pos-v"] = 1.0
        # if token.pos_ == "ADV":
        #     features["pos-adv"] = 1.0


class TokenFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["tok["+str(relative_idx)+"]="+tokens[current_idx+relative_idx]] = 1.0


class UppercaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if tokens[current_idx+relative_idx].isupper():
            features["uppercase["+str(relative_idx)+"]"] = 1.0


class TitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if tokens[current_idx+relative_idx].istitle():
            features["titlecase["+str(relative_idx)+"]"] = 1.0


class InitialTitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if current_idx+relative_idx==0 and tokens[current_idx+relative_idx].istitle():
            features["initialtitlecase["+str(relative_idx)+"]"] = 1.0


class PunctuationFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if PUNC_REPEAT_RE.match(tokens[current_idx+relative_idx]):
            features["punc["+str(relative_idx)+"]"] = 1.0


class DigitFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if DIGIT_RE.search(tokens[current_idx+relative_idx]):
            features["digit["+str(relative_idx)+"]"] = 1.0


class WordShapeFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        new_token = UPPERCASE_RE.sub("X", tokens[current_idx+relative_idx])
        new_token = LOWERCASE_RE.sub("x", new_token)
        new_token = DIGIT_RE.sub("0", new_token)
        features["shape["+str(relative_idx)+"]="+new_token] = 1.0


class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.feature_extractors = feature_extractors
        self.window_size = window_size
        self.features = {}

    def extract(self, tokens: Sequence[Token]) -> List[Dict[str, float]]:
        feature_list = []
        tokens_str = [t.text for t in tokens]
        for i in range(0, len(tokens)):
            for feature_extractor in self.feature_extractors:
                for j in range(-1*self.window_size, self.window_size+1):
                    if i+j >= 0 and i+j < len(tokens):
                        if not isinstance(feature_extractor, POSFeature):
                            feature_extractor.extract(tokens[i].text, i, j, tokens_str, self.features)
                        else:
                            feature_extractor.extract(tokens[i], i, j, tokens, self.features)
            feature_list.append(self.features)
            self.features = {}
        return feature_list


class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.feature_extractor = feature_extractor
        self.crfs_tagger = None
        self._encoder = encoder

    @property
    def encoder(self) -> EntityEncoder:
        return self._encoder

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        crfs_trainer = pycrfsuite.Trainer(algorithm, verbose=False)
        crfs_trainer.set_params(params)
        for doc in docs:
            tokens = [token for token in doc]
            # features = self.feature_extractor.extract([token.text for token in tokens])
            features = self.feature_extractor.extract(tokens)
            encoded_labels = self._encoder.encode(tokens)
            crfs_trainer.append(features, encoded_labels)
        crfs_trainer.train(path)
        self.crfs_tagger = pycrfsuite.Tagger()
        self.crfs_tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        if self.crfs_tagger is None:
            raise ValueError("Tagger has not been trained")

        entities = []
        tokens = [token for token in doc]
        predicted_bilou_labels = self.predict_labels(tokens)
        entities.extend(self.decode_bilou(predicted_bilou_labels, tokens, doc))
        doc.ents = entities[:]
        return doc

    def predict_labels(self, tokens: Sequence[Token]) -> List[str]:
        # features = self.feature_extractor.extract([token.text for token in tokens])
        features = self.feature_extractor.extract(tokens)
        return self.crfs_tagger.tag(features)

    def decode_bilou(self, labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
        start_index = tokens[0].i
        end_index = start_index
        spans = []  # List[Span]

        # presumption: labels is not empty
        prev_entity_type = "O" if labels[0] == "O" else labels[0].split("-")[1]
        # add a dummy "O" to the end so the real last token in the labels can be evaluated
        labels.append("O")
        for j in range(1, len(labels)):
            curr_entity_type = "O" if labels[j] == "O" else labels[j].split("-")[1]
            curr_entity_label = "O" if labels[j] == "O" else labels[j].split("-")[0]
            if prev_entity_type == "O":
                start_index = tokens[j - 1].i + 1
                end_index = start_index
            elif curr_entity_label in ("I", "L") and curr_entity_type == prev_entity_type:
                end_index += 1
            else:
                spans.append(Span(doc, start_index, end_index + 1, prev_entity_type))
                start_index = end_index + 1
                end_index = start_index

            prev_entity_type = curr_entity_type
        return spans


class BILOUEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        spans = [] # List[str]
        # presumption: labels is not empty
        prev_entity_type = tokens[0].ent_type_
        prev_entity_label = tokens[0].ent_iob_

        for j in range(1, len(tokens)):
            curr_entity_type = tokens[j].ent_type_
            curr_entity_label = tokens[j].ent_iob_
            if prev_entity_type == "":
                spans.append("O")
            elif prev_entity_label == "B" and prev_entity_type != curr_entity_type:
                spans.append("U-"+prev_entity_type)
            elif prev_entity_label == "B":
                spans.append("B-"+prev_entity_type)
            elif prev_entity_label == "I" and prev_entity_type == curr_entity_type:
                spans.append("I-"+prev_entity_type)
            elif prev_entity_label == "I":
                spans.append("L-"+prev_entity_type)

            prev_entity_type = curr_entity_type
            prev_entity_label = curr_entity_label

        # add the last token
        if prev_entity_label == "":
            spans.append("O")
        elif prev_entity_label == "B":
            spans.append("U-"+prev_entity_type)
        elif prev_entity_label == "I":
            spans.append("L-"+prev_entity_type)
        return spans

    def get_encoder_name(self):
        return "BILOU encoder"


class BIOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        spans = []  # List[str]

        for token in tokens:
            if token.ent_type_ == "":
                spans.append("O")
            else:
                spans.append(token.ent_iob_ + "-" + token.ent_type_)
        return spans

    def get_encoder_name(self):
        return "BIO encoder"


class IOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        spans = []
        for token in tokens:
            if token.ent_type_ == "":
                spans.append("O")
            else:
                spans.append("I-"+token.ent_type_)
        return spans

    def get_encoder_name(self):
        return "IO encoder"


def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> ScoringCounts:
    test_dict = defaultdict(list)
    ref_dict = defaultdict(list)
    true_pos = []; false_pos = []; false_neg = []
    for test_doc in test_docs:
        test_ents = test_doc.ents
        for test_ent in test_ents:
            test_dict[test_ent.label_].append(test_ent)
    for ref_doc in reference_docs:
        ref_ents = ref_doc.ents
        for ref_ent in ref_ents:
            ref_dict[ref_ent.label_].append(ref_ent)

    # type is ignored (typed=false)
    if not typed:
        for i in range(len(test_docs)):
            test_ents = test_docs[i].ents
            ref_ents = reference_docs[i].ents
            for test_ent in test_ents:
                for ref_ent in ref_ents:
                    if test_ent.start == ref_ent.start and test_ent.end == ref_ent.end:
                        # add true positive to the list
                        true_pos.append(ScoringEntity(tuple(ref_ent.text.split()), ""))
                        # remove the correct prediction from false positive
                        test_ents = [ent for ent in test_ents if ent!=ref_ent]
                        # remove the correct prediction from false negative
                        ref_ents = [ent for ent in ref_ents if ent!=ref_ent]
            for ent in test_ents:
                false_pos.append(ScoringEntity(tuple(ent.text.split()), ""))
            for ent in ref_ents:
                false_neg.append(ScoringEntity(tuple(ent.text.split()), ""))
    # if typed=true, type is taking into account
    else:
        for label in set(list(ref_dict.keys()) + list(test_dict.keys())):
            for i in range(len(test_docs)):
                test_ents = test_docs[i].ents
                ref_ents = reference_docs[i].ents
                for test_ent in test_ents:
                    for ref_ent in ref_ents:
                        if span_equal(test_ent, ref_ent) and test_ent.label_ == label:
                            # add true positive to the list
                            true_pos.append(ScoringEntity(tuple(ref_ent.text.split()), ref_ent.label_))
                            # remove true positive entity from the false positive dict (test_dict)
                            test_dict[label] = [item for item in test_dict[label] if not span_equal(item, test_ent)]
                            # remove true positive entity from the false negative dict (ref_dict)
                            ref_dict[label] = [item for item in ref_dict[label] if not span_equal(item, ref_ent)]
            # add false positive to the list
            for ent in test_dict[label]:
                false_pos.append(ScoringEntity(tuple(ent.text.split()), ent.label_))
            # add false negative to the list
            for ent in ref_dict[label]:
                false_neg.append(ScoringEntity(tuple(ent.text.split()), ent.label_))

    if true_pos:
        true_pos_counter = Counter(true_pos)
    else:
        true_pos_counter = Counter()
    if false_pos:
        false_pos_counter = Counter(false_pos)
    else:
        false_pos_counter = Counter()
    if false_neg:
        false_neg_counter = Counter(false_neg)
    else:
        false_neg_counter = Counter()
    return ScoringCounts(true_pos_counter, false_pos_counter, false_neg_counter)


def span_equal(obj1, obj2):
    """
    test if two Span objects have the same label, text value, start and end position
    :param obj1: first Span obj
    :param obj2: second Span obj
    :return: true if two Span objects have the same label, text value, start and end position
    """
    return obj1.start == obj2.start and obj1.end==obj2.end and obj1.label_==obj2.label_ and obj1.text == obj2.text


def main(train_file: str, test_file: str):
    # best setting: word vector of scaling factor 1 and brown cluster plus prefixes [8,12,16,20]

    with open(train_file) as infile:
        lines = infile.readlines()

    mentions = defaultdict(lambda: defaultdict(int))
    for line in lines:
        json_line = json.loads(line)
        train_doc = ingest_json_document(json_line, NLP)
        if train_doc:
            for ent in train_doc.ents:
                label = ent.label_
                mention = ent.text
                mentions[label][mention] += 1

    # read data from test file
    with open(test_file) as infile:
        lines = infile.readlines()

    for line in lines:
        json_line = json.loads(line)
        test_doc = ingest_json_document(json_line, NLP)
        if test_doc:
            for ent in test_doc.ents:
                label = ent.label_
                mention = ent.text
                mentions[label][mention] += 1

    with open("test_output.csv", "w", newline="") as outf:
        writer = csv.writer(outf, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for label in mentions:
            for mention in mentions[label]:
                count = mentions[label][mention]
                if count > 0:
                    mention = mention.replace("\n", " ")
                    row = [label, mention, count]
                    writer.writerow(row)


if __name__ == "__main__":
    main("../annotated_data/train.jsonl", "../annotated_data/test.jsonl")