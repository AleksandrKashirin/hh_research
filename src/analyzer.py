r"""Researcher: collect statistics, predict salaries etc.

------------------------------------------------------------------------

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (c) 2020 Kapitanov Alexander

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT
WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND
PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE
DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR
OR CORRECTION.

------------------------------------------------------------------------
"""

# Authors       : Alexander Kapitanov
# ...
# Contacts      : <empty>
# License       : GNU GENERAL PUBLIC LICENSE

import re
from typing import Dict, List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns


class Analyzer:
    def __init__(self, save_csv: bool = False):
        self.save_csv = save_csv
        # try:
        #     nltk.download("stopwords")
        # except:
        #     print(r"[INFO] You have downloaded stopwords!")

    @staticmethod
    def find_top_words_from_keys(keys_list: List) -> pd.Series:
        """Find most used words into description of vacancies.

        Parameters
        ----------
        keys_list : list
            List of sentences from keywords of vacancies.

        Returns
        -------
        pd.Series
            List of sorted keywords.

        """
        # Create a list of keys for all vacancies
        lst_keys = []
        for keys_elem in keys_list:
            for el in keys_elem:
                if el != "":
                    lst_keys.append(re.sub("'", "", el.lower()))

        # Unique keys and their counter
        set_keys = set(lst_keys)
        # Dict: {Key: Count}
        dct_keys = {el: lst_keys.count(el) for el in set_keys}
        # Sorted dict
        srt_keys = dict(sorted(dct_keys.items(), key=lambda x: x[1], reverse=True))
        # Return pandas series
        return pd.Series(srt_keys, name="Keys")

    @staticmethod
    def find_top_words_from_description(desc_list: List) -> pd.Series:
        """Find most used words into description of vacancies.

        Parameters
        ----------
        desc_list : list
            List of sentences from vacancy description.

        Returns
        -------
        pd.Series
            List of sorted words from descriptions.

        """
        words_ls = " ".join([re.sub(" +", " ", re.sub(r"\d+", "", el.strip().lower())) for el in desc_list])
        # Find all words
        words_re = re.findall("[a-zA-Z]+", words_ls)
        # Filter words with length < 3
        words_l2 = [el for el in words_re if len(el) > 2]
        # Unique words
        words_st = set(words_l2)
        # Remove 'stop words'
        try:
            _ = nltk.corpus.stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        finally:
            stop_words = set(nltk.corpus.stopwords.words("english"))

        # XOR for dictionary
        words_st ^= stop_words
        words_st ^= {"amp", "quot"}
        # Dictionary - {Word: Counter}
        words_cnt = {el: words_l2.count(el) for el in words_st}
        # Pandas series
        return pd.Series(dict(sorted(words_cnt.items(), key=lambda x: x[1], reverse=True)))

    def prepare_df(self, vacancies: Dict) -> pd.DataFrame:
        """Prepare data frame and save results

        Parameters
        ----------
        vacancies: dict
            Dict of parsed vacancies.

        """

        # Create pandas dataframe
        df = pd.DataFrame.from_dict(vacancies)
        # Print some info from data frame
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df[df["Salary"]][["Employer", "From", "To"]][0:15])
        # Save to file
        if self.save_csv:
            print("\n\n[INFO]: Save dataframe to file...")
            df.to_csv(rf"hh_results.csv", index=False)
        return df

    def analyze_df(self, df: pd.DataFrame):
        """Load data frame and analyze results

        """
        sns.set()
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(df[df["Salary"]][0:7])

        # print("\nNumber of vacancies: {}".format(df["Ids"].count()))
        # print("\nВакансии с максимальной годовой стоимостью работника: ")
        # print(df.iloc[df[["From", "To"]].idxmax()])
        # print("\nВакансии с минимальной годовой стоимостью работника: ")
        # print(df.iloc[df[["From", "To"]].idxmin()])

        # print("\n[ИНФО]: Описание таблицы стоимости работника")
        # df_stat = df[["From", "To"]].describe().applymap(np.int32)
        # print(df_stat.iloc[list(range(4)) + [-1]])

        # print('\n[ИНФО]: Усредненная статистика:')
        # comb_ft = np.nanmean(df[df["Salary"]][["From", "To"]].to_numpy(), axis=1)
        # print("Описание данных по стоимости работника:")
        # print("Минимальная    : %d" % np.min(comb_ft))
        # print("Максимальная    : %d" % np.max(comb_ft))
        # print("Средняя   : %d" % np.mean(comb_ft))
        # print("Медиана : %d" % np.median(comb_ft))

        # print("\nMost frequently used words [Keywords]:")
        # most_keys = self.find_top_words_from_keys(df["Keys"].to_list())
        # print(most_keys[:12])

        # print("\nMost frequently used words [Description]:")
        # most_words = self.find_top_words_from_description(df["Description"].to_list())
        # print(most_words[:12])

        print("\n[ИНФО]: Построение графиков. Закройте окно с графиками для продолжения...")
        fz = plt.figure("Графики стоимостей работника", figsize=(12, 8))
        fz.add_subplot(2, 2, 1)
        plt.title("От / До: Диаграмма размаха")
        sns.boxplot(data=df[["From", "To"]].dropna() / 1000, width=0.4)
        plt.ylabel("Стоимость x 1000 [РУБ]")

        fz.add_subplot(2, 2, 2)
        plt.title("От / До: Диаграмма рассеяния")
        sns.swarmplot(data=df[["From", "To"]].dropna() / 1000, size=6)
        plt.ylabel("Стоимость x 1000 [РУБ]")

        fz.add_subplot(2, 2, 3)
        plt.title("От: Распределение")
        sns.histplot(df["From"].dropna() / 1000, bins=20, color="C0", kde=True)
        plt.grid(True)
        plt.xlabel("Стоимость x 1000 [РУБ]")
        plt.xlim([df["From"].min() / 1000, df["From"].max() / 1000])
        plt.yticks([], [])

        fz.add_subplot(2, 2, 4)
        plt.title("До: Распределение")
        sns.histplot(df["To"].dropna() / 1000, bins=20, color="C1", kde=True)
        plt.grid(True)
        plt.xlim([df["To"].min() / 1000, df["To"].max() / 1000])
        plt.xlabel("Стоимость x 1000 [РУБ]")
        plt.yticks([], [])
        plt.tight_layout()
        plt.show()

    def analyze_and_save_results(self, df, output_filename='salary_analysis.csv'):
        """Загружает DataFrame, анализирует результаты и сохраняет в CSV"""
        sns.set()

        # Создаем словарь для хранения результатов анализа
        analysis_results = {
            'Метрика': [],
            'Значение': []
        }

        # Добавляем базовую статистику
        analysis_results['Метрика'].append('Количество вакансий')
        analysis_results['Значение'].append(df["URL"].count())

        # Максимальные значения
        max_from_idx = df["From"].idxmax()
        max_to_idx = df["To"].idxmax()
        analysis_results['Метрика'].extend([
            'Максимальная стоимость "От" (руб)',
            'Название вакансии (макс. От)',
            'Максимальная стоимость "До" (руб)',
            'Название вакансии (макс. До)'
        ])
        analysis_results['Значение'].extend([
            df.loc[max_from_idx, "From"],
            df.loc[max_from_idx, "Employer"],
            df.loc[max_to_idx, "To"],
            df.loc[max_to_idx, "Employer"]
        ])

        # Минимальные значения
        min_from_idx = df[df["From"] > 0]["From"].idxmin()  # Игнорируем нулевые значения
        min_to_idx = df[df["To"] > 0]["To"].idxmin()  # Игнорируем нулевые значения
        analysis_results['Метрика'].extend([
            'Минимальная стоимость "От" (руб)',
            'Название вакансии (мин. От)',
            'Минимальная стоимость "До" (руб)',
            'Название вакансии (мин. До)'
        ])
        analysis_results['Значение'].extend([
            df.loc[min_from_idx, "From"],
            df.loc[min_from_idx, "Employer"],
            df.loc[min_to_idx, "To"],
            df.loc[min_to_idx, "Employer"]
        ])

        # Статистика по зарплатам с русскими названиями
        stats_mapping = {
            'count': 'количество',
            'mean': 'среднее',
            'std': 'стандартное отклонение',
            'min': 'минимум',
            'max': 'максимум'
        }

        # Вычисляем статистику только для ненулевых значений
        df_stat = df[df["Salary"]][["From", "To"]].describe().applymap(np.int32)

        for stat, stat_ru in stats_mapping.items():
            analysis_results['Метрика'].extend([
                f'Стоимость "От" ({stat_ru})',
                f'Стоимость "До" ({stat_ru})'
            ])
            analysis_results['Значение'].extend([
                df_stat.loc[stat, 'From'],
                df_stat.loc[stat, 'To']
            ])

        # Усредненная статистика с русскими названиями
        salary_data = df[df["Salary"]][["From", "To"]]
        comb_ft = np.nanmean(salary_data.to_numpy(), axis=1)

        analysis_results['Метрика'].extend([
            'Минимальная стоимость (среднее)',
            'Максимальная стоимость (среднее)',
            'Средняя стоимость по всем вакансиям',
            'Медианная стоимость по всем вакансиям'
        ])
        analysis_results['Значение'].extend([
            int(np.min(comb_ft)),
            int(np.max(comb_ft)),
            int(np.mean(comb_ft)),
            int(np.median(comb_ft))
        ])

        # Создаем DataFrame с результатами и сохраняем в CSV
        results_df = pd.DataFrame(analysis_results)
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"\n[ИНФО]: Результаты анализа сохранены в файл {output_filename}")

        # Выводим результаты в консоль для проверки
        print("\nКраткий обзор результатов:")
        for metric, value in zip(analysis_results['Метрика'], analysis_results['Значение']):
            print(f"{metric}: {value}")

        print("\n[ИНФО]: Анализ завершен!")

    def analyze_and_save_results1(self, df, output_filename='salary_analysis.csv'):
        """Load data frame, analyze results and save to CSV"""
        sns.set()

        # Создаем словарь для хранения результатов анализа
        analysis_results = {
            'Метрика': [],
            'Значение': []
        }

        # Добавляем базовую статистику
        analysis_results['Метрика'].append('Количество вакансий')
        analysis_results['Значение'].append(df["URL"].count())

        # Максимальные значения
        max_from_idx = df["From"].idxmax()
        max_to_idx = df["To"].idxmax()
        analysis_results['Метрика'].extend([
            'Максимальная зарплата "От" (руб)',
            'Название вакансии (макс. От)',
            'Максимальная зарплата "До" (руб)',
            'Название вакансии (макс. До)'
        ])
        analysis_results['Значение'].extend([
            df.loc[max_from_idx, "From"],
            df.loc[max_from_idx, "Employer"],
            df.loc[max_to_idx, "To"],
            df.loc[max_to_idx, "Employer"]
        ])

        # Минимальные значения
        min_from_idx = df["From"].idxmin()
        min_to_idx = df["To"].idxmin()
        analysis_results['Метрика'].extend([
            'Минимальная зарплата "От" (руб)',
            'Название вакансии (мин. От)',
            'Минимальная зарплата "До" (руб)',
            'Название вакансии (мин. До)'
        ])
        analysis_results['Значение'].extend([
            df.loc[min_from_idx, "From"],
            df.loc[min_from_idx, "Employer"],
            df.loc[min_to_idx, "To"],
            df.loc[min_to_idx, "Employer"]
        ])

        # Статистика по зарплатам
        df_stat = df[["From", "To"]].describe().applymap(np.int32)
        for stat in ['count', 'mean', 'std', 'min', 'max']:
            analysis_results['Метрика'].extend([
                f'From {stat}',
                f'To {stat}'
            ])
            analysis_results['Значение'].extend([
                df_stat.loc[stat, 'From'],
                df_stat.loc[stat, 'To']
            ])

        # Усредненная статистика
        comb_ft = np.nanmean(df[df["Salary"]][["From", "To"]].to_numpy(), axis=1)
        analysis_results['Метрика'].extend([
            'Средняя минимальная зарплата',
            'Средняя максимальная зарплата',
            'Общая средняя зарплата',
            'Медианная зарплата'
        ])
        analysis_results['Значение'].extend([
            int(np.min(comb_ft)),
            int(np.max(comb_ft)),
            int(np.mean(comb_ft)),
            int(np.median(comb_ft))
        ])

        # Создаем DataFrame с результатами и сохраняем в CSV
        results_df = pd.DataFrame(analysis_results)
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"\n[ИНФО]: Результаты анализа сохранены в файл {output_filename}")

        # Выводим результаты в консоль для проверки
        print("\nКраткий обзор результатов:")
        for metric, value in zip(analysis_results['Метрика'], analysis_results['Значение']):
            print(f"{metric}: {value}")


if __name__ == "__main__":
    analyzer = Analyzer()
    print("\n[INFO]: Analyze dataframe...")
    df = pd.read_csv('/home/aleksandr/Work/hh_research/data/продавец/hh_results.csv')
    analyzer.analyze_df(df)
    analyzer.analyze_and_save_results(df)
    print("[INFO]: Done! Exit()")
