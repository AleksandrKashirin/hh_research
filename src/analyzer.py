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

import csv
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns


class Analyzer:
    def __init__(self, save_csv: bool = False, profession_name: str = "unknown"):
        self.save_csv = save_csv
        self.profession_name = profession_name
        # try:
        #     nltk.download("stopwords")
        # except:
        #     print(r"[INFO] You have downloaded stopwords!")

    def get_output_directory(self):
        """Создает и возвращает путь к директории для сохранения результатов"""
        # Безопасное имя папки
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", self.profession_name)
        # Создаем путь к директории
        output_dir = os.path.join("data", safe_name)
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

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
        # Если в vacancies остался ключ "stats", удаляем его
        if "stats" in vacancies:
            vacancies.pop("stats", None)

        # Create pandas dataframe
        df = pd.DataFrame.from_dict(vacancies)
        
        # Проверяем, что DataFrame не пустой и содержит необходимые столбцы
        if not df.empty and all(col in df.columns for col in ["Employer", "From", "To", "Salary"]):
            # Print some info from data frame
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(df[df["Salary"]][["Employer", "From", "To"]][0:15])
        else:
            print("\n[ПРЕДУПРЕЖДЕНИЕ]: DataFrame пустой или не содержит ожидаемых столбцов")
        
        # Save to file в каталог профессии, а не в корень проекта
        if self.save_csv and not df.empty:
            output_dir = self.get_output_directory()
            csv_path = os.path.join(output_dir, "hh_results.csv")
            print(f"\n\n[INFO]: Сохранение DataFrame в {csv_path}")
            df.to_csv(csv_path, index=False, sep=';')
        
        return df

    def analyze_df(self, df: pd.DataFrame):
        """Load data frame and analyze results"""
        sns.set()
        
        # Проверяем, что DataFrame не пустой
        if df.empty:
            print("\n[ПРЕДУПРЕЖДЕНИЕ]: DataFrame пуст, невозможно построить графики.")
            return
        
        # Проверяем, что DataFrame содержит необходимые столбцы
        if not all(col in df.columns for col in ["From", "To"]):
            print("\n[ПРЕДУПРЕЖДЕНИЕ]: DataFrame не содержит столбцы From и To, невозможно построить графики.")
            return
        
        # Проверяем, что существуют непустые значения для построения графиков
        if df["From"].dropna().empty and df["To"].dropna().empty:
            print("\n[ПРЕДУПРЕЖДЕНИЕ]: Нет данных о зарплатах для построения графиков.")
            return

        print("\n[ИНФО]: Построение графиков. Закройте окно с графиками для продолжения...")
        
        # Создаем каталог для сохранения графиков, если он еще не существует
        output_dir = self.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        fz = plt.figure("Графики стоимостей работника", figsize=(12, 8))
        
        # График 1: Диаграмма размаха
        fz.add_subplot(2, 2, 1)
        plt.title("От / До: Диаграмма размаха")
        sns.boxplot(data=df[["From", "To"]].dropna() / 1000, width=0.4)
        plt.ylabel("Зарплата x 1000 [РУБ]")

        # График 2: Диаграмма рассеяния
        fz.add_subplot(2, 2, 2)
        plt.title("От / До: Диаграмма рассеяния")
        sns.swarmplot(data=df[["From", "To"]].dropna() / 1000, size=6)
        plt.ylabel("Зарплата x 1000 [РУБ]")

        # График 3: Распределение "От"
        fz.add_subplot(2, 2, 3)
        plt.title("От: Распределение")
        if not df["From"].dropna().empty:
            sns.histplot(df["From"].dropna() / 1000, bins=20, color="C0", kde=True)
            plt.grid(True)
            plt.xlabel("Зарплата x 1000 [РУБ]")
            plt.xlim([df["From"].min() / 1000, df["From"].max() / 1000])
        plt.yticks([], [])

        # График 4: Распределение "До"
        fz.add_subplot(2, 2, 4)
        plt.title("До: Распределение")
        if not df["To"].dropna().empty:
            sns.histplot(df["To"].dropna() / 1000, bins=20, color="C1", kde=True)
            plt.grid(True)
            plt.xlabel("Зарплата x 1000 [РУБ]")
            plt.xlim([df["To"].min() / 1000, df["To"].max() / 1000])
        plt.yticks([], [])
        
        # Улучшаем расположение элементов
        plt.tight_layout()
        
        # Сохраняем график в каталог профессии
        plot_path = os.path.join(output_dir, "salary_distribution.png")
        print(f"[ИНФО]: Сохранение графика в {plot_path}")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        
        # Показываем график (опционально, можно закомментировать для автоматического режима)
        # plt.show()
        
        # Также сохраняем DataFrame для этой профессии
        csv_path = os.path.join(output_dir, "hh_results.csv")
        print(f"[ИНФО]: Сохранение DataFrame в {csv_path}")
        df.to_csv(csv_path, index=False, sep=';')

    def analyze_and_save_results(self, df, output_filename=None, stats=None):
        """Загружает DataFrame, анализирует результаты и сохраняет в CSV"""
        sns.set()

        # Определяем путь к выходному файлу
        if output_filename is None:
            output_dir = self.get_output_directory()
            output_filename = os.path.join(output_dir, "salary_analysis.csv")

        print(f"[ИНФО]: Анализ будет сохранен в {output_filename}")
        
        # Создаем словарь для хранения результатов анализа с разделами
        analysis_results = {"Метрика": [], "Значение": []}

        # -------- РАЗДЕЛ: ОБЩАЯ ИНФОРМАЦИЯ --------
        # Добавляем заголовок раздела
        analysis_results["Метрика"].append("___ ОБЩАЯ ИНФОРМАЦИЯ ___")
        analysis_results["Значение"].append("")

        # Добавляем информацию о найденных и обработанных вакансиях, если доступно
        if stats:
            found_vacancies = stats.get("found_vacancies", "Н/Д")
            processed_vacancies = stats.get("processed_vacancies", "Н/Д")

            analysis_results["Метрика"].extend(["Всего найдено вакансий по запросу", "Успешно обработано вакансий"])
            analysis_results["Значение"].extend([found_vacancies, processed_vacancies])

        # Добавляем количество вакансий в DataFrame
        if not df.empty:
            vac_count = df["URL"].count()
            analysis_results["Метрика"].append("Количество вакансий в выборке")
            analysis_results["Значение"].append(vac_count)

        # Добавляем пустую строку как разделитель
        analysis_results["Метрика"].append("")
        analysis_results["Значение"].append("")

        # Если DataFrame пустой, сохраняем только базовую информацию
        if df.empty:
            print(f"\n[ПРЕДУПРЕЖДЕНИЕ]: DataFrame пустой, сохраняем только базовую статистику в {output_filename}")
            results_df = pd.DataFrame(analysis_results)
            results_df.to_csv(output_filename, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
            print(f"\n[ИНФО]: Базовая статистика сохранена в файл {output_filename}")
            return

        # -------- РАЗДЕЛ: РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ --------
        analysis_results["Метрика"].append("___ РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ ___")
        analysis_results["Значение"].append("")

        if "ProfessionName" in df.columns:
            # Подсчитываем количество вакансий по профессиям
            profession_counts = df["ProfessionName"].value_counts()
            print("\nРаспределение по профессиям:")
            print(profession_counts.head(10))  # Топ-10 профессий

            # Добавляем в результаты анализа
            analysis_results["Метрика"].append("-- Распределение по профессиям --")
            analysis_results["Значение"].append("")

            for profession, count in profession_counts.head(5).items():
                analysis_results["Метрика"].append(f"Профессия: {profession}")
                analysis_results["Значение"].append(count)

            # Пустая строка после блока
            analysis_results["Метрика"].append("")
            analysis_results["Значение"].append("")

        if "IndustryName" in df.columns:
            # Подсчитываем количество вакансий по отраслям
            industry_counts = df["IndustryName"].value_counts()
            print("\nРаспределение по отраслям:")
            print(industry_counts.head(10))  # Топ-10 отраслей

            # Добавляем в результаты анализа
            analysis_results["Метрика"].append("-- Распределение по отраслям --")
            analysis_results["Значение"].append("")

            for industry, count in industry_counts.head(5).items():
                analysis_results["Метрика"].append(f"Отрасль: {industry}")
                analysis_results["Значение"].append(count)

            # Пустая строка после блока
            analysis_results["Метрика"].append("")
            analysis_results["Значение"].append("")

        if "IndustryGroupName" in df.columns:
            # Подсчитываем количество вакансий по группам индустрий
            industry_group_counts = df["IndustryGroupName"].value_counts()
            print("\nРаспределение по группам индустрий:")
            print(industry_group_counts.head(10))  # Топ-10 групп индустрий

            # Добавляем в результаты анализа
            analysis_results["Метрика"].append("-- Распределение по группам индустрий --")
            analysis_results["Значение"].append("")

            for group, count in industry_group_counts.head(5).items():
                analysis_results["Метрика"].append(f"Группа индустрий: {group}")
                analysis_results["Значение"].append(count)

            # Пустая строка после блока
            analysis_results["Метрика"].append("")
            analysis_results["Значение"].append("")

        # -------- РАЗДЕЛ: ЭКСТРЕМАЛЬНЫЕ ЗНАЧЕНИЯ ЗАРПЛАТ --------
        analysis_results["Метрика"].append("___ ЭКСТРЕМАЛЬНЫЕ ЗНАЧЕНИЯ ЗАРПЛАТ ___")
        analysis_results["Значение"].append("")

        # Максимальные значения
        max_from_idx = df["From"].idxmax()
        max_to_idx = df["To"].idxmax()
        analysis_results["Метрика"].extend(
            [
                'Максимальная зарплата "От" (руб)',
                "Название вакансии (макс. От)",
                'Максимальная зарплата "До" (руб)',
                "Название вакансии (макс. До)",
            ]
        )
        analysis_results["Значение"].extend(
            [
                df.loc[max_from_idx, "From"],
                df.loc[max_from_idx, "Employer"],
                df.loc[max_to_idx, "To"],
                df.loc[max_to_idx, "Employer"],
            ]
        )

        # Пустая строка после блока
        analysis_results["Метрика"].append("")
        analysis_results["Значение"].append("")

        # Минимальные значения
        min_from_idx = df[df["From"] > 0]["From"].idxmin()  # Игнорируем нулевые значения
        min_to_idx = df[df["To"] > 0]["To"].idxmin()  # Игнорируем нулевые значения
        analysis_results["Метрика"].extend(
            [
                'Минимальная зарплата "От" (руб)',
                "Название вакансии (мин. От)",
                'Минимальная зарплата "До" (руб)',
                "Название вакансии (мин. До)",
            ]
        )
        analysis_results["Значение"].extend(
            [
                df.loc[min_from_idx, "From"],
                df.loc[min_from_idx, "Employer"],
                df.loc[min_to_idx, "To"],
                df.loc[min_to_idx, "Employer"],
            ]
        )

        # Пустая строка после блока
        analysis_results["Метрика"].append("")
        analysis_results["Значение"].append("")

        # -------- РАЗДЕЛ: СТАТИСТИКА ПО ЗАРПЛАТАМ --------
        analysis_results["Метрика"].append("___ СТАТИСТИКА ПО ЗАРПЛАТАМ ___")
        analysis_results["Значение"].append("")

        # Статистика по зарплатам с русскими названиями
        stats_mapping = {
            "count": "количество",
            "mean": "среднее",
            "std": "стандартное отклонение",
            "min": "минимум",
            "max": "максимум",
        }

        # Вычисляем статистику только для ненулевых значений
        df_stat = df[df["Salary"]][["From", "To"]].describe().applymap(np.int32)

        # Статистика "От"
        analysis_results["Метрика"].append('-- Статистика "От" --')
        analysis_results["Значение"].append("")

        for stat, stat_ru in stats_mapping.items():
            analysis_results["Метрика"].append(f'Зарплата "От" ({stat_ru})')
            analysis_results["Значение"].append(df_stat.loc[stat, "From"])

        # Пустая строка после блока
        analysis_results["Метрика"].append("")
        analysis_results["Значение"].append("")

        # Статистика "До"
        analysis_results["Метрика"].append('-- Статистика "До" --')
        analysis_results["Значение"].append("")

        for stat, stat_ru in stats_mapping.items():
            analysis_results["Метрика"].append(f'Зарплата "До" ({stat_ru})')
            analysis_results["Значение"].append(df_stat.loc[stat, "To"])

        # Пустая строка после блока
        analysis_results["Метрика"].append("")
        analysis_results["Значение"].append("")

        # -------- РАЗДЕЛ: СВОДНАЯ СТАТИСТИКА --------
        analysis_results["Метрика"].append("___ СВОДНАЯ СТАТИСТИКА ___")
        analysis_results["Значение"].append("")

        # Усредненная статистика с русскими названиями
        salary_data = df[df["Salary"]][["From", "To"]]
        comb_ft = np.nanmean(salary_data.to_numpy(), axis=1)

        analysis_results["Метрика"].extend(
            [
                "Минимальная зарплата (среднее)",
                "Максимальная зарплата (среднее)",
                "Средняя зарплата по всем вакансиям",
                "Медианная зарплата по всем вакансиям",
            ]
        )
        analysis_results["Значение"].extend(
            [int(np.min(comb_ft)), int(np.max(comb_ft)), int(np.mean(comb_ft)), int(np.median(comb_ft))]
        )

        # Создаем DataFrame с результатами и сохраняем в CSV
        results_df = pd.DataFrame(analysis_results)
        results_df.to_csv(output_filename, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL, sep=";")

        print(f"\n[ИНФО]: Результаты анализа сохранены в файл {output_filename}")

        # Выводим результаты в консоль для проверки
        print("\nКраткий обзор результатов:")
        for i, (metric, value) in enumerate(zip(analysis_results["Метрика"], analysis_results["Значение"])):
            # Пропускаем пустые строки и заголовки разделов для консольного вывода
            if metric and not metric.startswith("___") and not metric.startswith("--"):
                print(f"{metric}: {value}")

        print("\n[ИНФО]: Анализ завершен!")


if __name__ == "__main__":
    analyzer = Analyzer()
    print("\n[INFO]: Analyze dataframe...")
    df = pd.read_csv("/home/aleksandr/Work/hh_research/data/продавец/hh_results.csv")
    analyzer.analyze_df(df)
    analyzer.analyze_and_save_results(df)
    print("[INFO]: Done! Exit()")
