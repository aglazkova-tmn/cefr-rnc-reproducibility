import pandas as pd


def get_lists_of_mwe(orig_text, lemmatized_text):
    my_text = orig_text
    my_text = my_text.lower()

    data_mwe = pd.read_csv("data/mwe_all_levels.csv", quotechar='"')
    # списки паттернов для красивого вывода на печать правильной формы
    # должны подаваться на вход в отсортированном виде, сначала самые длинные.
    patterns_proper_forms = list(data_mwe["to_print"])
    # список лемматизированных паттернов
    patterns_lem = list(data_mwe["lemmatized"])
    # делаем паттерны с тильдой
    patterns_tilda = [f.replace(" ", "~") for f in patterns_lem]

    punkt = [" ", ",", ".", "!", "?", ";", '"', "»", "«"]

    # уровневые списки в формах для печати
    patterns_a1 = list(data_mwe[data_mwe["final_level"] == 1]["to_print"])
    patterns_a2 = patterns_a1 + list(data_mwe[data_mwe["final_level"] == 2]["to_print"])
    patterns_b1 = patterns_a2 + list(data_mwe[data_mwe["final_level"] == 3]["to_print"])
    patterns_b2 = patterns_b1 + list(data_mwe[data_mwe["final_level"] == 4]["to_print"])
    patterns_c1 = patterns_b2 + list(data_mwe[data_mwe["final_level"] == 5]["to_print"])

    # прагратические выражения
    data_pragmatikon = pd.read_csv("data/pragmatikon_all_levels.csv", quotechar="`")
    pragmatikon_all = list(data_pragmatikon["to_print"])
    pragmatikon_a1 = list(
        data_pragmatikon[data_pragmatikon["final_level"] == 1]["to_print"]
    )
    pragmatikon_a2 = pragmatikon_a1 + list(
        data_pragmatikon[data_pragmatikon["final_level"] == 2]["to_print"]
    )
    pragmatikon_b1 = pragmatikon_a2 + list(
        data_pragmatikon[data_pragmatikon["final_level"] == 3]["to_print"]
    )
    pragmatikon_b2 = pragmatikon_b1 + list(
        data_pragmatikon[data_pragmatikon["final_level"] == 4]["to_print"]
    )
    pragmatikon_c1 = pragmatikon_b2 + list(
        data_pragmatikon[data_pragmatikon["final_level"] == 5]["to_print"]
    )

    def replace_strings_with_patterns(my_list, patterns_tilda, patterns_proper_form):
        for i, s in enumerate(my_list):
            if s in patterns_tilda:
                pattern_idx = patterns_tilda.index(s)
                my_list[i] = patterns_proper_form[pattern_idx]
        return my_list

    # первый цикл - поиск с пунктуацией по прагматикону
    pragmatikon_in_text = []

    not_in_a1_pragmatikon = []
    not_in_a2_pragmatikon = []
    not_in_b1_pragmatikon = []
    not_in_b2_pragmatikon = []
    not_in_c1_pragmatikon = []

    for pragma in pragmatikon_all:
        index = my_text.find(pragma)
        if index != -1:
            start_index = my_text.find(pragma)
            finish_index = my_text.find(pragma) + len(pragma)
            last_index = len(my_text)
            if (my_text[start_index - 1] in punkt or start_index == 0) and (
                finish_index == last_index or my_text[finish_index] in punkt
            ):
                # print(pragma,'нашел')
                my_text = my_text.replace(pragma, "")
                pragmatikon_in_text.append(pragma)
                if pragma not in pragmatikon_a1:
                    not_in_a1_pragmatikon.append(pragma)
                if pragma not in pragmatikon_a2:
                    not_in_a2_pragmatikon.append(pragma)
                if pragma not in pragmatikon_b1:
                    not_in_b1_pragmatikon.append(pragma)
                if pragma not in pragmatikon_b2:
                    not_in_b2_pragmatikon.append(pragma)
                if pragma not in pragmatikon_c1:
                    not_in_c1_pragmatikon.append(pragma)

    lem_text = lemmatized_text
    # списки нечетких сочетаний: to do
    # print(fuzz.ratio('у меня аллергия', 'у Антона аллергия'))

    # Заменяем пробелы на ~ для каждой максимальной подстроки из patterns в my_text
    for pattern in patterns_lem:
        if pattern in lem_text:
            # проверка что начинаем с первого символа слова (этот время - тот время)
            start_index = lem_text.find(pattern)
            finish_index = lem_text.find(pattern) + len(pattern)
            last_index = len(lem_text)
            if (lem_text[start_index - 1] in punkt or start_index == 0) and (
                finish_index == last_index or lem_text[finish_index] in punkt
            ):
                # print("нашел", pattern)
                split_pattern = pattern.split()
                replace_pattern = "~".join(split_pattern)
                lem_text = (
                    lem_text[:start_index] + replace_pattern + lem_text[finish_index:]
                )
    splited = lem_text.split(" ")

    words = [f for f in splited if "~" not in f]
    mwes = [f for f in splited if "~" in f]

    splited_proper_forms = replace_strings_with_patterns(
        splited, patterns_tilda, patterns_proper_forms
    )
    index_dict = {
        word: splited_proper_forms.index(word) for word in splited_proper_forms
    }

    # объединяем список отдельных слов и выражений
    if mwes:
        mwe_proper_forms = [
            patterns_proper_forms[patterns_tilda.index(f)] for f in mwes
        ]
        not_in_a1_mwe = [f for f in mwe_proper_forms if f not in patterns_a1]
        not_in_a2_mwe = [f for f in mwe_proper_forms if f not in patterns_a2]
        not_in_b1_mwe = [f for f in mwe_proper_forms if f not in patterns_b1]
        not_in_b2_mwe = [f for f in mwe_proper_forms if f not in patterns_b2]
        not_in_c1_mwe = [f for f in mwe_proper_forms if f not in patterns_c1]
    else:
        mwe_proper_forms = []
        not_in_a1_mwe = []
        not_in_a2_mwe = []
        not_in_b1_mwe = []
        not_in_b2_mwe = []
        not_in_c1_mwe = []

    if mwes or pragmatikon_in_text:
        return (
            True,
            index_dict,
            words,
            mwe_proper_forms,
            pragmatikon_in_text,
            not_in_a1_mwe,
            not_in_a2_mwe,
            not_in_b1_mwe,
            not_in_b2_mwe,
            not_in_c1_mwe,
            not_in_a1_pragmatikon,
            not_in_a2_pragmatikon,
            not_in_b1_pragmatikon,
            not_in_b2_pragmatikon,
            not_in_c1_pragmatikon,
        )
    else:
        return False, index_dict, words
