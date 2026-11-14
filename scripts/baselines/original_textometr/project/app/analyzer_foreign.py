import re
import statistics
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from joblib import load, dump
from razdel import sentenize
from sklearn import linear_model
from .find_multiword_expressions import get_lists_of_mwe


class AnalyzerForeign:

    SYLLABLES = ["а", "е", "ё", "и", "о", "у", "ы", "ю", "я", "э"]

    MODAL_WORDS = [
        "хочется",
        "нужно",
        "надо",
        "кажется",
        "казаться",
        "пожалуй",
        "хотеть",
        "должный",
        "хотеться",
    ]

    COLUMNS_NEEDED_FOREIGN = [
        "inA2",
        "kellyC2",
        "simple850",
        "simple1000",
        "simple2000",
        "dale3000",
        "infr1000",
        "infr3000",
        "infr5000",
        "infr10000",
        "lex_abstract",
        "formula_smog",
        "mean_len_word",
        "median_len_word",
        "median_len_sentence",
        "mean_len_sentence",
        "percent_of_long_words",
        "mean_punct_per_sentence",
        "median_punct_per_sentence",
        "contentPOS",
        "kotoryi/words",
        "modal_verbs",
        "conj_adversative",
        "kotoryi/sentences",
        "passive",
        "A",
        "ADV",
        "ADVPRO",
        "ANUM",
        "APRO",
        "COM",
        "CONJ",
        "INTJ",
        "NUM",
        "PART",
        "PR",
        "S",
        "SPRO",
        "V",
        "наст",
        "непрош",
        "прош",
        "вин",
        "дат",
        "им",
        "пр",
        "род",
        "твор",
        "ед",
        "мн",
        "деепр",
        "изъяв",
        "инф",
        "пов",
        "прич",
        "кр",
        "полн",
        "притяж",
        "1-л",
        "2-л",
        "3-л",
        "жен",
        "муж",
        "сред",
        "несов",
        "сов",
        "действ",
        "страд",
        "неод",
        "од",
        "нп",
        "пе",
    ]

    GRAM_FEATURES = [
        "A",
        "ADV",
        "ADVPRO",
        "ANUM",
        "APRO",
        "COM",
        "CONJ",
        "INTJ",
        "NUM",
        "PART",
        "PR",
        "S",
        "SPRO",
        "V",
        "непрош",
        "прош",
        "им",
        "пр",
        "род",
        "твор",
        "деепр",
        "изъяв",
        "инф",
        "пов",
        "прич",
        "кр",
        "полн",
        "притяж",
        "1-л",
        "2-л",
        "3-л",
        "сред",
        "несов",
        "сов",
        "действ",
        "страд",
        "неод",
        "од",
        "прев",
        "срав",
        "муж",
        "жен",
        "ед",
        "мн",
    ]

    # Список потенциально сложных тем
    DIFFICULT_GRAM = [
        "A",
        "ADV",
        "ADVPRO",
        "ANUM",
        "APRO",
        "NUM",
        "SPRO",
        "им",
        "пр",
        "род",
        "твор",
        "деепр",
        "пов",
        "прич",
        "кр",
        "полн",
        "притяж",
        "несов",
        "сов",
        "modal_verbs",
        "passive",
    ]

    GR_FEATURES_NAMES = [
        "Прилагательные",
        "Наречия",
        "Наречия",
        "Порядковые числительные",
        "Прилагательные",
        "Числительные",
        "Местоимения",
        "Именительный падеж",
        "Предложный падеж",
        "Родительный падеж",
        "Творительный падеж",
        "Деепричастия",
        "Повелительное наклонение",
        "Причастия",
        "Краткие формы прилагательных и причастий",
        "Краткие формы прилагательных и причастий",
        "Притяжательные местоимения",
        "Вид глагола",
        "Вид глагола",
        "Модальные глаголы",
        "Пассивные формы",
    ]

    def __init__(self, mystem):
        self.mystem = mystem

        self.ridge = linear_model.Ridge(alpha=0.1)

        self.features = pd.read_csv("data/list_of_features_1207.csv")

        # словники
        self.slovnik_A1_list = self.__load("data/optima_a1.txt")
        self.slovnik_A2_list = self.__load("data/optima_a2.txt")
        self.slovnik_B1_list = self.__load("data/vocab_b1_optima_from_A2.txt")
        self.slovnik_B2_list = self.__load("data/vocab_b2_optima_from_A2.txt")
        self.slovnik_C1_list = self.__load("data/vocab_с1_optima_from_A2.txt")

        # списки kelly
        self.kelly_A1_list = self.__load("data/kelly_a1.txt")
        self.kelly_A2_list = self.__load("data/kelly_a2.txt")
        self.kelly_B1_list = self.__load("data/kelly_b1.txt")
        self.kelly_B2_list = self.__load("data/kelly_b2.txt")
        self.kelly_C1_list = self.__load("data/kelly_c1.txt")
        self.kelly_C2_list = self.__load("data/kelly_c2.txt")

        # списки частотных слов
        self.fr_100_list = self.__load("data/fr_100.txt")
        self.fr_300_list = self.__load("data/fr_300.txt")
        self.fr_500_list = self.__load("data/fr_500.txt")
        self.fr_1000_list = self.__load("data/fr_1000.txt")
        self.fr_3000_list = self.__load("data/fr_3000.txt")
        self.fr_5000_list = self.__load("data/fr_5000.txt")
        self.fr_10000_list = self.__load("data/fr_10000.txt")
        self.fr_more_than_5list = self.__load("data/fr_more_than_5ipm.txt")
        self.fr_spoken_list = self.__load("data/fr_spoken.txt")

        # списки слов
        self.simple_russian_850_list = self.__load("data/SimpleRussian850.txt")
        self.simple_russian_1000_list = self.__load("data/simple_russian.txt")
        self.simple_russian_2000_list = self.__load("data/SimpleRussian2000.txt")
        self.brown_russian_10000_list = self.__load("data/Brown10000.txt")
        self.dale_russian_3000_list = self.__load("data/DaleRussian3000.txt")
        self.stop_list = self.__load("data/stop_list.txt")

        # семантические списки
        self.lex_abstract_list = self.__load("data/lex_abstract.txt")

        # считали датафрейм
        self.corpus_rnc = pd.read_csv("data/freq_rnc.csv", quotechar="`")
        self.lemmas_list_rnc = list(self.corpus_rnc["lemma"])

        # создаем словарь и будем в него все складывать
        self.data_about_text = {}
        self.whole_lemmas_list = []
        self.noun_list = []
        self.bastard_list = []
        self.obsc_list = []
        self.razgovor_list = []
        self.names_list = []
        self.geo_name_list = []
        self.conj_adversative_list = []  # противительные союзы
        self.modal_words_list = []
        self.words_length_list = []
        self.number_of_syllables_list = []
        self.long_words_list = []  # слова более чем из 4 слогов
        self.long_words_len_list = []
        self.count_kotoryi = []
        self.count_content_pos = []
        self.count_passive = []
        self.whole_lemmas_minus_geo_and_stop = []
        self.count_all_verbs = []
        self.count_reflexive_verbs = []

        # Обучаем модель
        # x_train, y_train = self.features[AnalyzerForeign.COLUMNS_NEEDED_FOREIGN], self.features["level"]
        # self.ridge.fit(x_train.values, y_train.values)
        # dump(self.ridge, "data/model.joblib")

        # Загружаем уже обученную модель
        self.ridge = load("data/model.joblib")

    # считаем слоги и буквы
    def __count_syllables(self, element):
        i_text = element.get("text")
        i_text_syl_counter = 0
        for ii in i_text:
            if ii in AnalyzerForeign.SYLLABLES:
                i_text_syl_counter += 1
        if i_text_syl_counter == 0:
            i_text_syl_counter = 1
        self.number_of_syllables_list.append(i_text_syl_counter)
        self.words_length_list.append(len(i_text))
        if i_text_syl_counter >= 4:
            self.long_words_len_list.append(i_text_syl_counter)
            self.long_words_list.append(i_text)
        return True

    # чистимся от имен, геообъектов и бастардов
    def __clean_from_name_geo_bastard(self, element):
        self.whole_lemmas_list.append(element.get("analysis")[0]["lex"])
        gr_info = element.get("analysis")[0]["gr"]
        if "qual" in element.get("analysis")[0]:
            if element.get("analysis")[0]["qual"] == "bastard":
                self.bastard_list.append(element.get("text"))
        if gr_info.find("гео") > 0:
            self.geo_name_list.append(element.get("analysis")[0]["lex"])
        if gr_info.find("обсц") > 0:
            self.obsc_list.append(element.get("analysis")[0]["lex"])
        if gr_info.find("разг") > 0:
            self.razgovor_list.append(element.get("analysis")[0]["lex"])
        if (
            gr_info.find("имя") > 0
            or gr_info.find("фам") > 0
            or gr_info.find("отч") > 0
        ):
            self.names_list.append(element.get("analysis")[0]["lex"])
        if (
            element.get("analysis")[0]["lex"] == "но"
            or element.get("analysis")[0]["lex"] == "а"
            or element.get("analysis")[0]["lex"] == "однако"
            or element.get("analysis")[0]["lex"] == "зато"
        ):
            self.conj_adversative_list.append(element.get("analysis")[0]["lex"])
        if element.get("analysis")[0]["lex"] in AnalyzerForeign.MODAL_WORDS:
            self.modal_words_list.append(element.get("analysis")[0]["lex"])
        if element.get("analysis")[0]["lex"] == "который":
            self.count_kotoryi.append(element.get("analysis")[0]["lex"])
        return True

    # подсчет грам. информации
    def __count_gram(self, element):
        gr_info = element.get("analysis")[0]["gr"]
        gr_info = gr_info.replace(",", "<b>")
        gr_info = gr_info.replace("=", "<b>")
        gr_info = gr_info.split("<b>")
        for i in AnalyzerForeign.GRAM_FEATURES:
            if i in gr_info:
                self.dict_of_features[i] += 1
        if "S" in gr_info:
            self.noun_list.append(element.get("analysis")[0]["lex"])
        if "S" in gr_info or "V" in gr_info or "A" in gr_info or "ADV" in gr_info:
            self.count_content_pos.append(element.get("analysis")[0]["lex"])
        if "V" in gr_info:
            lemma_x = element.get("analysis")[0]["lex"]
            self.count_all_verbs.append(lemma_x)
            if lemma_x.endswith("ся") or lemma_x.endswith("сь"):
                self.count_reflexive_verbs.append(lemma_x)

        return True

    def __count_passive_form(self, element):
        if element[0].get("analysis") and element[1].get("analysis"):
            element0_gr = self.__get_gr_info(element[0])
            element1_gr = self.__get_gr_info(element[1])
            if element[0].get("analysis")[0]["lex"] == "быть" and "прош" in element0_gr:
                if "прич" in element1_gr:
                    self.count_passive.append(element[1].get("text"))
        return True

    # считаем пунктуацию по предложениям
    def __punctuation_per_sentence(self, element):
        list_punctuation_score = []
        punctuation = [",", "-", ":", ";", "—"]
        for i in self.sentences:
            counter = 0
            for ii in i:
                if ii in punctuation:
                    counter += 1
            list_punctuation_score.append(counter)
        return list_punctuation_score

    def __load(self, file_name):
        f = open(file_name, "r", encoding="utf_8")
        lines = f.readlines()
        f.close()
        return [line.replace("\n", "") for line in lines]

    def __get_gr_info(self, element):
        gr_info = element.get("analysis")[0]["gr"]
        gr_info = gr_info.replace(",", "<b>")
        gr_info = gr_info.replace("=", "<b>")
        gr_info = gr_info.split("<b>")
        return gr_info

    # Общий цикл просмотра анализа слов
    def __gram_analyze(self, element):
        for i in element:
            self.__count_syllables(i)
            if len(i.get("analysis")) > 0:
                self.__clean_from_name_geo_bastard(i)
                self.__count_gram(i)
        return True

    # Вычисляем процент слов из разных словников и частотных списков
    # todo: когда буду переписывать модель, унифицировать везде процент! Оставить тот который _100
    def __percent_of_known_words(self, element, list_of_words):
        if len(list_of_words) == 0 or len(element) == 0:
            return 0
        else:
            known_words = [w for w in element if w in list_of_words]
            percent = round((len(known_words) / len(element)), 2)
            return percent

    def __percent_of_known_words_100(self, element, list_of_words):
        if len(list_of_words) == 0 or len(element) == 0:
            return 0
        else:
            known_words = [w for w in element if w in list_of_words]
            percent = round((len(known_words) / len(element)) * 100)
            return percent

    def __count_freq_by_rnc(self, element):
        if element in self.lemmas_list_rnc:
            # если омонимы, берем cамый частотный
            if self.lemmas_list_rnc.count(element) > 1:
                idf = float(
                    list(self.corpus_rnc[self.corpus_rnc["lemma"] == element]["idf"])[0]
                )
            else:
                idf = float(
                    list(self.corpus_rnc[self.corpus_rnc["lemma"] == element]["idf"])[0]
                )
        else:
            idf = 0
        return idf

    def __get_frequency_bag(self, element):
        frequency_bag = dict()
        unique_lemmas = sorted(list(set(element)))
        for i in unique_lemmas:
            frequency_bag[i] = element.count(i)
        sorted_fr_bag = sorted(frequency_bag.items(), key=lambda x: x[1], reverse=True)
        return sorted_fr_bag

    def __clean_text(self, input_text):
        new_text = input_text.replace("­\n", "")
        new_text = new_text.replace("\n", " ")
        new_text = new_text.replace("•", "")
        new_text = new_text.replace("…", ".")
        new_text = new_text.replace("...", ".")
        new_text = new_text.replace("..", ".")
        new_text = new_text.replace("?.", "?")
        new_text = new_text.replace("!.", "!")
        new_text = new_text.replace("_", "")
        new_text = new_text.replace("!—", "! —")
        new_text = new_text.replace("?—", "? —")
        new_text = new_text.replace("\xad", "")
        return new_text

    def __sent_tokenize(self, this_text):
        return [sent.text for sent in sentenize(this_text)]

    def __clear_fields(self):
        self.data_about_text = {}
        self.whole_lemmas_list = []
        self.noun_list = []
        self.bastard_list = []
        self.obsc_list = []
        self.names_list = []
        self.geo_name_list = []
        self.conj_adversative_list = []  # противительные союзы
        self.modal_words_list = []
        self.words_length_list = []
        self.number_of_syllables_list = []
        self.long_words_list = []  # слова более чем из 4 слогов
        self.long_words_len_list = []
        self.count_kotoryi = []
        self.count_content_pos = []
        self.count_passive = []
        self.whole_lemmas_minus_geo_and_stop = []
        self.count_all_verbs = []
        self.count_reflexive_verbs = []
        return True

    def __first_check_len_text(self, element):
        # первая проверка текста - не слишком маленький
        if len(element) < 10:
            self.data_about_text["text_ok"] = False
            self.data_about_text["text_error_message"] = (
                "Введите текст на русском языке не менее 5 слов."
            )
            return self.data_about_text

        # Вторая проверка текста - не слишком большой
        if len(element) > 10000:
            self.data_about_text["text_ok"] = False
            self.data_about_text["text_error_message"] = (
                "Введите текст не более 10 000 знаков."
            )
            return self.data_about_text

        self.data_about_text["text_ok"] = True
        self.data_about_text["text_error_message"] = ""

        return self.data_about_text

    def __second_check_len_text(self, element):
        if len(element) < 5:
            self.data_about_text["text_ok"] = False
            self.data_about_text["text_error_message"] = (
                "Введите текст на русском языке не менее 5 слов."
            )
            return self.data_about_text

        self.data_about_text["text_ok"] = True
        self.data_about_text["text_error_message"] = ""
        return self.data_about_text

    def __set_numbers_about_foreign_text(self, clean_lemmas_list):
        self.all_words = len(self.whole_analyzed_text)
        self.all_len_words = [len(f) for f in self.whole_lemmas_list]
        self.all_syllables = sum(self.number_of_syllables_list)
        self.all_len_sentences = [len(f.split(" ")) for f in self.sentences]
        self.all_sentences = len(self.sentences)
        self.long_words = len(self.long_words_list)
        self.unique_lemmas_list = list(set(self.whole_lemmas_list))

        self.noun_unique_list = list(set(self.noun_list))  # список уникальных сущ.

        self.whole_lemmas_minus_geo_and_stop = [
            f
            for f in self.whole_lemmas_list
            if f not in self.geo_name_list
            and f not in self.names_list
            and f not in self.stop_list
        ]

        # всего слов в тексте
        self.dict_of_features["words"] = len(self.whole_analyzed_text)
        self.dict_of_features["characters"] = len(self.raw_text)
        # всего предложений в тексте
        self.dict_of_features["sentences"] = len(self.sentences)
        # средняя длина слова в тексте
        self.dict_of_features["mean_len_word"] = round(
            ((sum(self.words_length_list)) / self.all_words), 1
        )
        self.dict_of_features["median_len_word"] = statistics.median(self.all_len_words)
        self.dict_of_features["median_len_sentence"] = statistics.median(
            self.all_len_sentences
        )

        # средняя длина предложения в тексте
        self.dict_of_features["mean_len_sentence"] = round(
            (self.all_words / self.all_sentences), 1
        )
        self.dict_of_features["mean_len_word_in_syllables"] = round(
            (self.all_syllables / self.all_words), 1
        )
        self.dict_of_features["percent_of_long_words"] = round(
            (self.long_words / self.all_words), 2
        )

        ASL = self.all_words / self.all_sentences
        ASW = self.all_syllables / self.all_words

        formula_f_oborneva_genuine = round(206.835 - (60.1 * ASW) - (1.3 * ASL))

        self.dict_of_features["formula_flesh_oborneva_number"] = (
            formula_f_oborneva_genuine
        )

        # формула для средней школы
        FKGsis = 0.36 * ASL + 5.76 * ASW - 11.97

        self.dict_of_features["formula_fkg_secondary_sis"] = round(FKGsis, 1)

        # type-token ratio - number of types and the number of tokens -
        # lexical variety
        self.dict_of_features["tt_ratio"] = round(
            ((len(self.unique_lemmas_list) / len(self.whole_lemmas_list))), 2
        )

        self.dict_of_features["lex_density"] = round(
            len(self.count_content_pos) / len(self.whole_lemmas_list), 2
        )

        # Среднее количество модальных глаголов и противительных союзов
        # на предложение
        self.dict_of_features["conj_adversative"] = (
            len(self.conj_adversative_list) / self.all_sentences
        )
        self.dict_of_features["modal_verbs"] = (
            len(self.modal_words_list) / self.all_words
        )
        self.dict_of_features["kotoryi/words"] = len(self.count_kotoryi) / len(
            self.whole_lemmas_list
        )
        self.dict_of_features["kotoryi/sentences"] = (
            len(self.count_kotoryi) / self.all_sentences
        )
        self.dict_of_features["contentPOS"] = round(
            (len(self.count_content_pos) / len(self.whole_lemmas_list)), 2
        )
        self.dict_of_features["passive"] = len(self.count_passive)

        self.dict_of_features["unique_words"] = len(self.unique_lemmas_list)

        # Доля названий и бастардов
        self.dict_of_features["geo_rel_words"] = len(self.geo_name_list) / len(
            self.whole_lemmas_list
        )

        self.dict_of_features["names_rel_words"] = len(self.names_list) / len(
            self.whole_lemmas_list
        )

        LEXICAL_LISTS = {
            "inA1": self.slovnik_A1_list,
            "inA2": self.slovnik_A2_list,
            "inB1": self.slovnik_B1_list,
            "inB2": self.slovnik_B2_list,
            "inC1": self.slovnik_C1_list,
            "kellyA1": self.kelly_A1_list,
            "kellyA2": self.kelly_A2_list,
            "kellyB1": self.kelly_B1_list,
            "kellyB2": self.kelly_B2_list,
            "kellyC1": self.kelly_C1_list,
            "kellyC2": self.kelly_C2_list,
            "fr_1000": self.fr_1000_list,
            "fr_3000": self.fr_3000_list,
            "fr_5000": self.fr_5000_list,
            "fr_10000": self.fr_10000_list,
            "infr_more_than_5": self.fr_more_than_5list,
            "infr_spoken": self.fr_spoken_list,
            "simple850": self.simple_russian_850_list,
            "simple1000": self.simple_russian_1000_list,
            "simple2000": self.simple_russian_2000_list,
            "dale3000": self.dale_russian_3000_list,
        }

        # Доля слов, входящих в различные лексические списки
        for i in LEXICAL_LISTS:
            new_key = self.__percent_of_known_words(
                self.whole_lemmas_minus_geo_and_stop, LEXICAL_LISTS[i]
            )
            self.dict_of_features[i] = new_key

        # Доля абстрактных/конкретных сущ. от всех сущ.
        self.dict_of_features["lex_abstract"] = self.__percent_of_known_words(
            self.noun_unique_list, self.lex_abstract_list
        )

        return self.dict_of_features

    def start_foreign(self, raw_text):
        self.__clear_fields()

        self.raw_text = raw_text
        text = self.__clean_text(raw_text)

        # первая проверка текста на длину
        self.data_about_text = self.__first_check_len_text(text)

        if self.data_about_text["text_ok"] is False:
            return self.data_about_text

        self.sentences = self.__sent_tokenize(text)
        self.whole_analyzed_text = self.mystem.analyze(text)  # весь текст одним списком
        self.whole_lemmatized_text_str = " ".join(
            self.mystem.lemmatize(text)
        ).lower()  # весь текст строкой лемм
        analyzed_bigrams = list(
            nltk.bigrams(self.whole_analyzed_text)
        )  # биграммочки c грам инфой

        # создаем словарь со всеми данными из текста
        self.dict_of_features = defaultdict(int)

        for i in AnalyzerForeign.GRAM_FEATURES:
            self.dict_of_features[i] = 0

        # запускаем функцию со всеми грам. анализами
        self.__gram_analyze(self.whole_analyzed_text)

        for i in AnalyzerForeign.GRAM_FEATURES:
            self.dict_of_features[str(i) + "_rel_words"] = round(
                self.dict_of_features[i] / len(self.whole_lemmas_list), 4
            )

        # подсчитываем пассивные формы
        for i in analyzed_bigrams:
            self.__count_passive_form(i)

        # вторая проверка, не менее 5 слов
        self.data_about_text = self.__second_check_len_text(self.whole_lemmas_list)
        if self.data_about_text["text_ok"] is False:
            return self.data_about_text

        # меняем значения в словаре с простых счетчиков на процент
        # встречаемости в тексте (для модели)
        for i in AnalyzerForeign.GRAM_FEATURES:
            self.dict_of_features[i] = round(
                (self.dict_of_features[i] / len(self.whole_lemmas_list)), 2
            )

        self.dict_of_features["reflexive_verbs_rel_verbs"] = round(
            len(self.count_reflexive_verbs) / (len(self.count_all_verbs) + 0.01), 2
        )

        self.clean_lemmas_list = [
            f
            for f in self.whole_lemmas_list
            if f not in self.geo_name_list
            and f not in self.names_list
            and f not in self.bastard_list
            and f not in self.stop_list
        ]

        self.__set_numbers_about_foreign_text(self.clean_lemmas_list)

        self.whole_analyzed_text.clear()
        analyzed_bigrams.clear()

        # из словаря признаков делаем список
        features_for_test_text = []
        for ii in AnalyzerForeign.COLUMNS_NEEDED_FOREIGN:
            # словарь с признаками (если значение есть в словаре, добавляет, если нет - ставим ноль)
            features_for_test_text.append(self.dict_of_features.get(ii, 0))

        test_features_array = np.array(features_for_test_text)
        test_features_array = test_features_array.reshape(1, -1)

        prediction = self.__predict(test_features_array)

        self.__tell_me_about_text(prediction)

        for key, value in self.dict_of_features.items():
            if key not in self.data_about_text:
                self.data_about_text[key] = value

        return self.data_about_text

    # Делаем предсказание
    def __predict(self, y):
        prediction = self.ridge.predict(y)
        if prediction < 0:
            prediction = 0
        return prediction

    # Принимает на вход уровень текста и выдает статистику
    def __tell_me_about_text(self, element):
        prediction = element

        if prediction > 0:
            prediction = round(prediction[0], 1)
        if prediction < 0.5:
            prediction = 0.5
        if prediction > 7:
            prediction = 7

        # округленный до целых уровень, чтобы потом анализировать по
        # средним значениям для этого уровня
        level_int = round(prediction)
        if level_int > 6:
            level_int = 6

        # уровень * 1.4, чтобы растянуть шкалу [0, 7] до [0, 10]
        level_for_scale = round((prediction * 1.4), 1)

        self.data_about_text["level_number"] = prediction
        self.data_about_text["level_int"] = level_int
        self.data_about_text["level_for_scale"] = level_for_scale

        f_by_levels = [
            self.features.iloc[:, :][self.features["level"] == 0],
            self.features.iloc[:, :][self.features["level"] == 1],
            self.features.iloc[:, :][self.features["level"] == 2],
            self.features.iloc[:, :][self.features["level"] == 3],
            self.features.iloc[:, :][self.features["level"] == 4],
            self.features.iloc[:, :][self.features["level"] == 5],
            self.features.iloc[:, :][self.features["level"] == 6],
        ]

        slovnik_by_levels = [
            self.slovnik_A1_list,
            self.slovnik_A2_list,
            self.slovnik_B1_list,
            self.slovnik_B2_list,
            self.slovnik_C1_list,
        ]
        kelly_by_levels = [
            self.kelly_A1_list,
            self.kelly_A2_list,
            self.kelly_B1_list,
            self.kelly_B2_list,
            self.kelly_C1_list,
        ]

        # начинаем анализ
        # слов в тексте
        self.data_about_text["words"] = self.dict_of_features["words"]

        self.data_about_text["characters"] = len(self.raw_text)

        # предложений
        self.data_about_text["sentences"] = self.dict_of_features["sentences"]

        self.data_about_text["unique_words"] = self.dict_of_features["unique_words"]

        self.data_about_text["tt_ratio"] = self.dict_of_features["tt_ratio"]

        # проверка на мат
        if len(self.obsc_list) > 0:
            self.data_about_text["obsc_check"] = True
            self.data_about_text["obsc_check_warning"] = (
                "Внимание! Текст может содержать нецензурную лексику."
            )

        # частотный словарь по тексту
        self.data_about_text["frequency_bag"] = self.__get_frequency_bag(
            self.whole_lemmas_list
        )

        # 1. Считаем ключевые слова по tf/idf
        bag_tf_idf = dict()
        # берем только основные части речи
        for item in self.count_content_pos:
            if (
                len(item) > 2
                and item not in self.bastard_list
                and item not in self.names_list
                # вернуть если поедут ключевые слова and item not in self.slovnik_A1_list
                and self.whole_lemmas_list.count(item) > 1
            ):
                idf = self.__count_freq_by_rnc(item)
                bag_tf_idf[item] = self.whole_lemmas_list.count(item) - (20 * idf)

        sorted_bag = sorted(bag_tf_idf.items(), key=lambda x: x[1], reverse=True)

        if len(self.raw_text) > 2500:
            self.data_about_text["key_words"] = [f[0] for f in sorted_bag[:20]]
        else:
            self.data_about_text["key_words"] = [f[0] for f in sorted_bag[:10]]

        # анализ многословных выражений в стороннем модуле:
        # на вход подаём сырой текст, на выход получаем уровневые списки
        # прагматикона и многословных выражений
        if get_lists_of_mwe(self.raw_text, self.whole_lemmatized_text_str):
            result = list(
                get_lists_of_mwe(self.raw_text, self.whole_lemmatized_text_str)
            )
            mwe_in_text = result[0]
            dict_of_indexes = result[1]
            one_words = result[2]
            if mwe_in_text is True:
                mwe_proper_forms = result[3]
                pragmatikon_in_text = result[4]
                not_in_a1_mwe = result[5]
                not_in_a2_mwe = result[6]
                self.data_about_text["out_of_A2_mwe"] = len(not_in_a2_mwe)
                not_in_b1_mwe = result[7]
                self.data_about_text["out_of_B1_mwe"] = len(not_in_b1_mwe)
                not_in_b2_mwe = result[8]
                self.data_about_text["out_of_B2_mwe"] = len(not_in_b2_mwe)
                not_in_c1_mwe = result[9]
                self.data_about_text["out_of_C1_mwe"] = len(not_in_c1_mwe)
                not_in_a1_pragmatikon = result[10]
                not_in_a2_pragmatikon = result[11]
                self.data_about_text["out_of_A2_pragmatikon"] = len(
                    not_in_a2_pragmatikon
                )
                not_in_b1_pragmatikon = result[12]
                self.data_about_text["out_of_B1_pragmatikon"] = len(
                    not_in_b1_pragmatikon
                )
                not_in_b2_pragmatikon = result[13]
                self.data_about_text["out_of_B2_pragmatikon"] = len(
                    not_in_b2_pragmatikon
                )
                not_in_c1_pragmatikon = result[14]
                self.data_about_text["out_of_C1_pragmatikon"] = len(
                    not_in_c1_pragmatikon
                )

        # Списки с минимумов
        self.data_about_text["inA1"] = round(self.dict_of_features["inA1"] * 100)
        self.data_about_text["inA2"] = round(self.dict_of_features["inA2"] * 100)
        self.data_about_text["inB1"] = round(self.dict_of_features["inB1"] * 100)
        self.data_about_text["inB2"] = round(self.dict_of_features["inB2"] * 100)
        self.data_about_text["inC1"] = round(self.dict_of_features["inC1"] * 100)

        self.data_about_text["not_inA1"] = sorted(
            list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if len(f) > 1
                        and f in one_words
                        and f not in self.stop_list
                        and f not in self.slovnik_A1_list
                    ]
                )
            ),
            key=lambda x: dict_of_indexes[x],
        )

        self.data_about_text["not_inA2"] = sorted(
            list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if len(f) > 1
                        and f in one_words
                        and f not in self.stop_list
                        and f not in self.slovnik_A2_list
                    ]
                )
            ),
            key=lambda x: dict_of_indexes[x],
        )

        self.data_about_text["not_inB1"] = sorted(
            list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if len(f) > 1
                        and f in one_words
                        and f not in self.stop_list
                        and f not in self.slovnik_B1_list
                    ]
                )
            ),
            key=lambda x: dict_of_indexes[x],
        )

        self.data_about_text["not_inB2"] = sorted(
            list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if len(f) > 1
                        and f in one_words
                        and f not in self.stop_list
                        and f not in self.slovnik_B2_list
                    ]
                )
            ),
            key=lambda x: dict_of_indexes[x],
        )

        self.data_about_text["not_inC1"] = sorted(
            list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if len(f) > 1
                        and f in one_words
                        and f not in self.stop_list
                        and f not in self.slovnik_C1_list
                    ]
                )
            ),
            key=lambda x: dict_of_indexes[x],
        )

        if mwe_in_text is True:
            if len(mwe_proper_forms) > 0:
                self.data_about_text["not_inA1"] = sorted(
                    self.data_about_text["not_inA1"] + not_in_a1_mwe,
                    key=lambda x: dict_of_indexes[x],
                )
                self.data_about_text["not_inA2"] = sorted(
                    self.data_about_text["not_inA2"] + not_in_a2_mwe,
                    key=lambda x: dict_of_indexes[x],
                )
                self.data_about_text["not_inB1"] = sorted(
                    self.data_about_text["not_inB1"] + not_in_b1_mwe,
                    key=lambda x: dict_of_indexes[x],
                )
                self.data_about_text["not_inB2"] = sorted(
                    self.data_about_text["not_inB2"] + not_in_b2_mwe,
                    key=lambda x: dict_of_indexes[x],
                )
                self.data_about_text["not_inC1"] = sorted(
                    self.data_about_text["not_inC1"] + not_in_c1_mwe,
                    key=lambda x: dict_of_indexes[x],
                )
            self.data_about_text["not_inA1"] += not_in_a1_pragmatikon
            self.data_about_text["not_inA2"] += not_in_a2_pragmatikon
            self.data_about_text["not_inB1"] += not_in_b1_pragmatikon
            self.data_about_text["not_inB2"] += not_in_b2_pragmatikon
            self.data_about_text["not_inC1"] += not_in_c1_pragmatikon
            self.data_about_text["mwes"] = mwe_proper_forms
            self.data_about_text["pragmatikon"] = pragmatikon_in_text
        else:
            self.data_about_text["mwes"] = ""
            self.data_about_text["pragmatikon"] = ""
        # Если уровень высокий, не выводим слова, не входящие в списки низких уровней: это примерно все:)

        self.data_about_text["infr5000"] = round(
            self.dict_of_features["infr5000"] * 100
        )

        if level_int < 4:
            # Самые полезные слова
            cool_words = list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if f not in slovnik_by_levels[(level_int - 1)]
                        and (
                            f in self.fr_3000_list
                            or f in kelly_by_levels[level_int]
                            or self.whole_lemmas_list.count(f) > 2
                        )
                    ]
                )
            )

            if len(cool_words) > 20:
                bag_cool_words = {}
                for item in cool_words:
                    idf = self.__count_freq_by_rnc(item)
                    bag_cool_words[item] = idf
                sorted_cool_bag = sorted(
                    bag_cool_words.items(), key=lambda x: x[1], reverse=True
                )
                cool_words = [f[0] for f in sorted_cool_bag[:20]]

            self.data_about_text["cool_words"] = cool_words

            # Избавиться от этих слов
            self.data_about_text["rare_words"] = list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if (
                            f not in cool_words
                            and f not in self.geo_name_list
                            and f not in self.names_list
                            and f not in slovnik_by_levels[level_int]
                            and f not in kelly_by_levels[level_int]
                            and f not in self.data_about_text["key_words"]
                            and (f not in self.fr_10000_list or f in self.bastard_list)
                        )
                    ]
                )
            )

            # нет в словнике есть в Келли
            self.data_about_text["cool_but_not_in_slovnik"] = list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if (
                            f not in slovnik_by_levels[level_int]
                            and f in kelly_by_levels[level_int]
                            and f in self.fr_3000_list
                        )
                    ]
                )
            )

        else:
            # Самые полезные слова
            cool_words_more_4 = list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if (
                            f not in slovnik_by_levels[3]
                            and (
                                f in slovnik_by_levels[4]
                                or f in self.fr_5000_list
                                or f in kelly_by_levels[4]
                                or self.whole_lemmas_list.count(f) > 3
                            )
                        )
                    ]
                )
            )

            if len(cool_words_more_4) > 20:
                bag_cool_words_more_4 = {}
                for item in cool_words_more_4:
                    idf = self.__count_freq_by_rnc(item)
                    bag_cool_words_more_4[item] = idf
                sorted_cool_bag_more_4 = sorted(
                    bag_cool_words_more_4.items(), key=lambda x: x[1], reverse=True
                )
                cool_words_more_4 = [f[0] for f in sorted_cool_bag_more_4[:20]]

            self.data_about_text["cool_words"] = cool_words_more_4

            # Избавиться от этих слов
            rare_words_more_4 = list(
                set(
                    [
                        f
                        for f in self.clean_lemmas_list
                        if (
                            f not in cool_words_more_4
                            and f not in self.geo_name_list
                            and f not in self.names_list
                            and f not in slovnik_by_levels[4]
                            and f not in self.data_about_text["key_words"]
                            and (f not in self.fr_10000_list or f in self.bastard_list)
                        )
                    ]
                )
            )
            self.data_about_text["rare_words"] = rare_words_more_4

        gram_complex = []

        for i in AnalyzerForeign.DIFFICULT_GRAM:
            compare = self.dict_of_features[i] / (
                np.mean(f_by_levels[level_int][i] + 0.00000001)
            )
            if compare > 2:
                gram_complex.append(
                    AnalyzerForeign.GR_FEATURES_NAMES[
                        AnalyzerForeign.DIFFICULT_GRAM.index(i)
                    ]
                )

        self.data_about_text["gram_complex"] = list(set(gram_complex))

        # Проверка и заполнение пустых ячеек нулями
        for key in self.data_about_text:
            if not self.data_about_text[key]:
                self.data_about_text[key] = 0

        return self.data_about_text
