r"""Vacancy finder

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

import hashlib
import os
import pickle
import random  # Добавляем импорт random для случайных задержек
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache")


def random_delay(base=1.0, variation=0.5):
    """Добавляет случайную задержку для предотвращения блокировки"""
    delay = base + random.uniform(-variation, variation)
    time.sleep(max(0.5, delay))  # Минимум 0.5 секунды


def create_robust_session():
    """Создает надежную сессию requests с настройками повторных попыток и таймаутов"""
    session = requests.Session()

    # Настраиваем стратегию повторных попыток с большими задержками
    retry_strategy = Retry(
        total=35,  # Уменьшаем количество попыток
        backoff_factor=2,  # Увеличиваем фактор задержки
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    # Увеличиваем таймаут
    session.timeout = 15

    # Добавляем более реалистичные заголовки браузера
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "DNT": "1",
        }
    )

    return session


class DataCollector:
    r"""Researcher parameters

    Parameters
    ----------
    exchange_rates : dict
        Dict of exchange rates: RUR, USD, EUR.

    """

    __API_BASE_URL = "https://api.hh.ru/vacancies/"
    __DICT_KEYS = (
        "Ids",
        "URL",
        "Employer",
        "Name",
        "Salary",
        "From",
        "To",
        "Experience",
        "Schedule",
        "Keys",
        "Description",
        "City",
        "ProfessionID",
        "ProfessionName",
        "IndustryID",
        "IndustryName",
        "IndustryGroupID",
        "IndustryGroupName",
    )

    def __init__(self, exchange_rates: Optional[Dict]):
        self._rates = exchange_rates

    @staticmethod
    def clean_tags(html_text: str) -> str:
        """Remove HTML tags from the string

        Parameters
        ----------
        html_text: str
            Input string with tags

        Returns
        -------
        result: string
            Clean text without HTML tags

        """
        pattern = re.compile("<.*?>")
        return re.sub(pattern, "", html_text)

    @staticmethod
    def __convert_gross(is_gross: bool) -> float:
        return 1 if is_gross else 1.1494

    def get_employer_info(self, employer_id, session=None):
        """Получить информацию о работодателе, включая отрасль"""
        if not session:
            session = create_robust_session()

        # Используем кэш для работодателей
        if not hasattr(self, "_employer_cache"):
            self._employer_cache = {}

        # Проверяем кэш
        if employer_id in self._employer_cache:
            return self._employer_cache[employer_id]

        # Формируем URL
        url = f"https://api.hh.ru/employers/{employer_id}"

        # Делаем запрос
        try:
            response = session.get(url)
            response.raise_for_status()
            employer = response.json()

            # Сохраняем в кэш
            self._employer_cache[employer_id] = employer
            return employer
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ]: Не удалось получить информацию о работодателе {employer_id}: {str(e)}")
            return None

    def get_vacancy(self, vacancy_id: str):
        # Создаем сессию
        session = create_robust_session()

        # Формируем URL
        url = f"{self.__API_BASE_URL}{vacancy_id}"

        # Пробуем получить данные с повторными попытками
        max_attempts = 5
        attempt = 0

        while attempt < max_attempts:
            try:
                response = session.get(url)
                response.raise_for_status()  # Проверяем успешность ответа
                vacancy = response.json()

                if vacancy and not vacancy.get("errors") and not vacancy.get("captcha"):
                    break
            except Exception as e:
                print(
                    f"[ПРЕДУПРЕЖДЕНИЕ]: Попытка {attempt+1}/{max_attempts} получения вакансии {vacancy_id} не удалась: {str(e)}"
                )

            attempt += 1
            time.sleep(1 + attempt)  # Увеличиваем задержку с каждой попыткой

        if attempt == max_attempts:
            print(f"[ОШИБКА]: Не удалось получить вакансию {vacancy_id} после {max_attempts} попыток")
            return None

        # Проверяем наличие обязательного слова в названии вакансии
        vacancy_name = vacancy.get("name", "").lower()
        if hasattr(self, "contains") and self.contains:
            if self.contains.lower() not in vacancy_name:
                return None

        # Extract salary
        salary = vacancy.get("salary")

        # Calculate salary:
        # Get salary into {RUB, USD, EUR} with {Gross} parameter and
        # return a new salary in RUB.
        from_to = {"from": None, "to": None}
        if salary:
            is_gross = vacancy["salary"].get("gross")
            for k, v in from_to.items():
                if vacancy["salary"][k] is not None:
                    _value = self.__convert_gross(is_gross)
                    from_to[k] = int(_value * salary[k] / self._rates[salary["currency"]])

        profession_id = None
        profession_name = None
        if vacancy.get("professional_roles") and len(vacancy["professional_roles"]) > 0:
            profession_id = vacancy["professional_roles"][0].get("id")
            profession_name = vacancy["professional_roles"][0].get("name")

        industry_id = None
        industry_name = None
        industry_group_id = None
        industry_group_name = None

        if vacancy.get("employer") and vacancy["employer"].get("id"):
            employer_id = vacancy["employer"]["id"]

            # Используем кэш для работодателей, чтобы не делать повторные запросы
            if not hasattr(self, "_employer_cache"):
                self._employer_cache = {}

            # Проверяем кэш
            if employer_id in self._employer_cache:
                employer_info = self._employer_cache[employer_id]
            else:
                # Создаем сессию
                session = create_robust_session()

                # Получаем информацию о работодателе
                try:
                    response = session.get(f"https://api.hh.ru/employers/{employer_id}")
                    response.raise_for_status()
                    employer_info = response.json()

                    # Сохраняем в кэш
                    self._employer_cache[employer_id] = employer_info
                except Exception as e:
                    print(f"[ПРЕДУПРЕЖДЕНИЕ]: Не удалось получить информацию о работодателе {employer_id}: {str(e)}")
                    employer_info = None

            # Извлекаем информацию об отрасли
            if employer_info and "industries" in employer_info and employer_info["industries"]:
                industry_id = employer_info["industries"][0].get("id")
                industry_name = employer_info["industries"][0].get("name")

                # Получаем информацию о группе индустрии
                if industry_id and "." in industry_id:
                    # Извлекаем часть ID до точки
                    industry_group_id = industry_id.split(".")[0]

                    # Если у нас нет кэша индустрий, получаем их
                    industries = self.get_industries()

                    # Ищем группу индустрии по ID
                    for group in industries:
                        if group.get("id") == industry_group_id:
                            industry_group_name = group.get("name")
                            break

        vacancy_url = "https://hh.ru/vacancy/" + str(vacancy_id)
        # Create pages tuple
        return (
            vacancy_id,
            vacancy_url,
            vacancy.get("name", ""),
            vacancy.get("employer", {}).get("name", ""),
            salary is not None,
            from_to["from"],
            from_to["to"],
            vacancy.get("experience", {}).get("name", ""),
            vacancy.get("schedule", {}).get("name", ""),
            [el["name"] for el in vacancy.get("key_skills", [])],
            self.clean_tags(vacancy.get("description", "")),
            vacancy.get("area", {}).get("name", ""),
            profession_id,
            profession_name,
            industry_id,
            industry_name,
            industry_group_id,
            industry_group_name,
        )

    def initialize_industries(self):
        """Инициализирует кэш индустрий при первом использовании"""
        if not hasattr(self, "_industry_groups_map"):
            industries = self.get_industries()
            self._industry_groups_map = {group["id"]: group["name"] for group in industries}

    @staticmethod
    def __encode_query_for_url(query: Optional[Dict]) -> str:
        # Создаем копию запроса, чтобы не изменять оригинал
        query_copy = query.copy() if query else {}

        # Обрабатываем параметр professional_roles отдельно
        if "professional_roles" in query_copy:
            roles = query_copy.pop("professional_roles")
            roles_params = "&".join([f"professional_role={r}" for r in roles])

            # Обрабатываем остальные параметры
            other_params = urlencode(query_copy, encoding="utf-8")

            # Объединяем параметры
            return roles_params + ("&" + other_params if other_params else "")

        # Если нет professional_roles, просто кодируем весь запрос
        return urlencode(query_copy, encoding="utf-8")

    def get_industries(self):
        """Получить список всех отраслей HeadHunter"""
        # Используем кэш для индустрий
        if hasattr(self, "_industries_cache"):
            return self._industries_cache

        session = create_robust_session()
        url = "https://api.hh.ru/industries"

        try:
            response = session.get(url)
            response.raise_for_status()
            industries = response.json()

            # Сохраняем в кэш
            self._industries_cache = industries
            return industries
        except Exception as e:
            print(f"[ПРЕДУПРЕЖДЕНИЕ]: Не удалось получить список отраслей: {str(e)}")
            return []

    def collect_vacancies(self, query: Optional[Dict], refresh: bool = False, num_workers: int = 1) -> Dict:
        """Parse vacancy JSON: get vacancy name, salary, experience etc.

        Parameters
        ----------
        query : dict
            Search query params for GET requests.
        refresh :  bool
            Refresh cached data
        num_workers :  int
            Number of workers for threading.

        Returns
        -------
        dict
            Dict of useful arguments from vacancies
        """
        # Извлекаем специальные параметры
        self.contains = query.pop('contains', None)  # Извлекаем параметр contains
        max_vacancies = query.pop('max_vacancies', None)  # Извлекаем max_vacancies
        
        if num_workers is None or num_workers < 1:
            num_workers = 1

        url_params = self.__encode_query_for_url(query)

        # Создаем надежную сессию с повторными попытками
        session = create_robust_session()

        # Проверяем кэш
        cache_name: str = url_params
        cache_hash = hashlib.md5(cache_name.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, cache_hash)
        try:
            if not refresh:
                print(f"[INFO]: Получение результатов из кэша! Включите опцию refresh для обновления данных.")
                cached_result = pickle.load(open(cache_file, "rb"))
                # Если в кэшированных данных нет статистики, добавляем приблизительные значения
                if "stats" not in cached_result:
                    cached_result["stats"] = {
                        "found_vacancies": "Данные из кэша (неизвестно)",
                        "processed_vacancies": len(cached_result.get("Ids", []))
                    }
                return cached_result
        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"[INFO]: Кэш не найден или поврежден. Будет выполнен новый запрос к API.")
            pass

        # Проверяем количество страниц с использованием надежной сессии
        try:
            target_url = self.__API_BASE_URL + "?" + url_params
            print(f"[INFO]: Запрос к API: {target_url}")
            
            # Пробуем получить данные с повторными попытками
            response = None
            data = None
            for attempt in range(3):
                try:
                    response = session.get(target_url, timeout=10)
                    response.raise_for_status()  # Проверяем успешность ответа
                    data = response.json()
                    break
                except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                    print(f"[ПРЕДУПРЕЖДЕНИЕ]: Попытка {attempt+1}/3 не удалась: {str(e)}")
                    if attempt == 2:  # Последняя попытка
                        raise
                    time.sleep(2 * (attempt + 1))  # Увеличиваем задержку с каждой попыткой
            
            if not data:
                print("[ОШИБКА]: Не удалось получить данные от API после нескольких попыток.")
                return {key: [] for key in self.__DICT_KEYS}
                
            num_pages = data.get("pages", 0)
            found_vacancies = data.get("found", 0)  # Общее количество найденных вакансий
            print(f"[INFO]: Найдено страниц: {num_pages}")
            print(f"[INFO]: Всего найдено вакансий: {found_vacancies}")
            
            # Если нет страниц или данных, возвращаем пустой результат
            if num_pages == 0 or "items" not in data:
                print("[ПРЕДУПРЕЖДЕНИЕ]: Не найдено данных. Проверьте параметры запроса.")
                empty_result = {key: [] for key in self.__DICT_KEYS}
                empty_result["stats"] = {
                    "found_vacancies": 0,
                    "processed_vacancies": 0
                }
                return empty_result
                
            # Собираем ID вакансий из первой страницы
            ids = []
            ids.extend(x["id"] for x in data["items"])
            
            # Получаем остальные страницы - здесь убираем ограничение в 20 страниц
            for idx in range(1, num_pages):
                try:
                    print(f"[INFO]: Получение страницы {idx}/{num_pages}...")
                    response = session.get(target_url, params={"page": idx}, timeout=10)
                    response.raise_for_status()
                    page_data = response.json()
                    
                    if "items" in page_data:
                        new_ids = [x["id"] for x in page_data["items"]]
                        ids.extend(new_ids)
                        print(f"[INFO]: Добавлено {len(new_ids)} вакансий с страницы {idx}")
                        
                    # Проверяем ограничение по количеству вакансий
                    if max_vacancies and len(ids) >= max_vacancies:
                        ids = ids[:max_vacancies]
                        print(f"[INFO]: Достигнут лимит в {max_vacancies} вакансий")
                        break
                        
                    # Добавляем задержку между запросами страниц
                    time.sleep(random.uniform(0.5, 1.5))
                    
                except Exception as e:
                    print(f"[ПРЕДУПРЕЖДЕНИЕ]: Ошибка при получении страницы {idx}: {str(e)}")
                    time.sleep(2.0)  # Увеличенная задержка при ошибке
                    continue
            
            print(f"[INFO]: Всего найдено {len(ids)} ID вакансий. Начинаем получение подробных данных...")
            
            # Собираем данные о вакансиях с использованием многопоточности
            jobs_list = []
            
            # Функция для получения вакансии с повторными попытками
            def get_vacancy_with_retry(vacancy_id):
                for retry in range(3):
                    try:
                        result = self.get_vacancy(vacancy_id)
                        return result
                    except Exception as e:
                        print(f"[ПРЕДУПРЕЖДЕНИЕ]: Ошибка при получении вакансии {vacancy_id} (попытка {retry+1}/3): {str(e)}")
                        time.sleep(1 + retry)
                return None
            
            # Используем ThreadPoolExecutor для параллельной обработки
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Создаем список задач
                futures = [executor.submit(get_vacancy_with_retry, vac_id) for vac_id in ids]
                
                # Отображаем прогресс выполнения
                for i, future in enumerate(tqdm(
                    futures, 
                    desc="Получение данных через API HH",
                    ncols=100,
                    total=len(ids),
                )):
                    try:
                        vacancy = future.result()
                        if vacancy:
                            jobs_list.append(vacancy)
                    except Exception as e:
                        print(f"[ПРЕДУПРЕЖДЕНИЕ]: Не удалось обработать задачу #{i}: {str(e)}")
            
            processed_vacancies = len(jobs_list)
            print(f"[INFO]: Успешно получено {processed_vacancies} вакансий из {len(ids)}")
            
            # Проверяем, есть ли результаты
            if not jobs_list:
                print("[ПРЕДУПРЕЖДЕНИЕ]: Нет данных для обработки после фильтрации.")
                empty_result = {key: [] for key in self.__DICT_KEYS}
                empty_result["stats"] = {
                    "found_vacancies": found_vacancies,
                    "processed_vacancies": 0
                }
                return empty_result
                
            # Транспонируем список кортежей в словарь по ключам
            unzipped_list = list(zip(*jobs_list))
            result = {}
            
            for idx, key in enumerate(self.__DICT_KEYS):
                result[key] = unzipped_list[idx]
                
            # Добавляем статистику
            result["stats"] = {
                "found_vacancies": found_vacancies,
                "processed_vacancies": processed_vacancies
            }
                
            # Сохраняем результаты в кэш
            try:
                # Проверяем, существует ли директория кэша
                if not os.path.exists(CACHE_DIR):
                    os.makedirs(CACHE_DIR)
                    
                pickle.dump(result, open(cache_file, "wb"))
                print(f"[INFO]: Результаты сохранены в кэш: {cache_file}")
            except Exception as e:
                print(f"[ПРЕДУПРЕЖДЕНИЕ]: Не удалось сохранить результаты в кэш: {str(e)}")
                
            return result
            
        except Exception as e:
            print(f"[ОШИБКА]: Не удалось получить данные от API: {str(e)}")
            empty_result = {key: [] for key in self.__DICT_KEYS}
            empty_result["stats"] = {
                "found_vacancies": 0,
                "processed_vacancies": 0
            }
            return empty_result

if __name__ == "__main__":
    dc = DataCollector(exchange_rates={"USD": 0.01264, "EUR": 0.01083, "RUR": 1.00000})

    vacancies = dc.collect_vacancies(
        query={"text": "FPGA", "area": 1, "per_page": 50},
        # refresh=True
    )
    print(vacancies["Employer"])
