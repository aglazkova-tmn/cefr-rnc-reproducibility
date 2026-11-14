import ru_core_news_lg
import re
from razdel import sentenize
import pandas as pd


class AnalyzerSyntaxFeatures:
    def __init__(self, mystem):
        self.mystem = mystem
        self.nlp = ru_core_news_lg.load()

    def __clean_text(self, input_text):
        new_text = input_text.replace("­\n", "")
        new_text = new_text.replace("\n", " ")
        new_text = new_text.replace("•", "")
        new_text = new_text.replace("\xad", "")
        return new_text

    def __sent_tokenize(self, this_text):
        return [sent.text for sent in sentenize(this_text)]

    def tree_height(self, root):
        if not list(root.children):
            return 0
        else:
            r = 1 + max(self.tree_height(x) for x in root.children)
            return r

    def count_clauses(self, sentroles):
        clauses = 1
        clause_marker = ["acl", "ccomp", "relcl", "acl:relcl", "acl:cleft"]
        for role in sentroles:
            if role in clause_marker:
                clauses += 1
        return clauses

    def return_list_of_synt_roles(self, list_of_sentences):
        counter = 1
        results = []
        counter_root_verb = 0
        counter_subject_noun = 0
        counter_complex_sent = 0
        counter_clauses_number = []
        counter_max_depth = []

        for i in list_of_sentences:
            doc = self.nlp(i)

            subject_pos = None
            root_pos = None
            roles = []
            sochinennoe = 0

            for token in doc:
                roles.append(token.dep_)

                if token.dep_ == "ROOT":
                    root_pos = token.pos_
                    if token.pos_ == "VERB":
                        counter_root_verb += 1
                    children_roles = list(f.dep_ for f in token.children)
                    sochinennoe += children_roles.count("conj")

                if "subj" in token.dep_:
                    subject_pos = token.pos_
                    if token.pos_ == "NOUN":
                        counter_subject_noun += 1

                if token.dep_ in ["acl", "ccomp", "relcl"]:
                    counter_complex_sent += 1

            clause_number = self.count_clauses(roles) + sochinennoe
            counter_clauses_number.append(clause_number)
            root = [token for token in doc if token.dep_ == "ROOT"][0]
            depth = self.tree_height(root)
            counter_max_depth.append(depth)

            results.append(
                {
                    "sentence": i,
                    "subject_pos": subject_pos,
                    "root_pos": root_pos,
                    "roles": roles,
                    "max_depth": depth,
                    "clause_number": clause_number,
                }
            )
            # print(f"Обработано {counter} предложений из {len(list_of_sentences)}")
            counter += 1

        # print("Разметка завершена")
        # print(results)

        roles_rate_per_text = {}
        all_roles = [role for result in results for role in result["roles"]]
        total_words = sum(len(sentence.split()) for sentence in list_of_sentences)
        role_counts = {
            role: all_roles.count(role)
            for role in set(all_roles)
            # / total_words for role in set(all_roles)
        }

        unique_roles = set(role_counts.keys())
        for role in unique_roles:
            roles_rate_per_text[f"role_{role}_rel_text"] = round(
                role_counts.get(role, 0) / total_words, 2
            )

        results_for_text = {}
        results_for_text["root_verbs_rel_sentence"] = round(
            counter_root_verb / len(list_of_sentences), 2
        )
        results_for_text["subject_nouns_rel_sentence"] = round(
            counter_subject_noun / len(list_of_sentences), 2
        )

        results_for_text["complex_markers_rel_sentence"] = round(
            counter_complex_sent / len(list_of_sentences), 2
        )
        results_for_text["roles_rates"] = roles_rate_per_text

        results_for_text["average_clauses_number"] = round(
            sum(counter_clauses_number) / len(list_of_sentences), 2
        )
        results_for_text["average_max_depth"] = round(
            sum(counter_max_depth) / len(list_of_sentences), 2
        )
        results_for_text["max_depth"] = max(counter_max_depth)

        results_for_text["max_clauses_number"] = max(counter_clauses_number)

        # print("Подсчет частоты ролей завершен", results_for_text)

        return results_for_text

    def start_syntax(self, input_text):
        clean_text = self.__clean_text(input_text)
        list_of_sentences = self.__sent_tokenize(clean_text)
        parsed_results = self.return_list_of_synt_roles(list_of_sentences)

        role_rates = parsed_results.pop("roles_rates", {})
        for role, rate in role_rates.items():
            parsed_results[role] = rate

        return parsed_results
