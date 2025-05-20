import os
import sys
import traceback
from typing import List, Optional, Dict

from src.analyzer import Analyzer
from src.currency_exchange import Exchanger
from src.data_collector import DataCollector
from src.parser import Settings
from src.predictor import Predictor

SETTINGS_PATH = "settings.json"

class MultiProcessor:
    """Класс для обработки нескольких профессий"""
    
    def __init__(self, professions_file: str = "professions.txt", config_path: str = SETTINGS_PATH):
        self.professions = self.load_professions(professions_file)
        self.config_path = config_path
        self.exchanger = Exchanger(config_path)
        self.settings = None
        self.init_settings()
        
    def load_professions(self, professions_file: str) -> List[str]:
        """Загружает список профессий из файла"""
        try:
            if not os.path.exists(professions_file):
                print(f"[ПРЕДУПРЕЖДЕНИЕ]: Файл {professions_file} не найден. Используются профессии по умолчанию.")
                return ["Машинист конвейера", "Обувщик"]
                
            with open(professions_file, 'r', encoding='utf-8') as f:
                professions = [line.strip() for line in f if line.strip()]
                
            if not professions:
                print("[ПРЕДУПРЕЖДЕНИЕ]: Файл профессий пуст. Используются профессии по умолчанию.")
                return ["Машинист конвейера", "Обувщик"]
                
            return professions
        except Exception as e:
            print(f"[ОШИБКА]: Не удалось загрузить профессии из файла {professions_file}: {str(e)}")
            return ["Машинист конвейера", "Обувщик"]
        
    def init_settings(self):
        """Инициализация настроек"""
        self.settings = Settings(self.config_path, no_parse=True)
        
        # Обновляем курсы валют если нужно
        if not any(self.settings.rates.values()) or self.settings.update:
            print("[INFO]: Trying to get exchange rates from remote server...")
            self.exchanger.update_exchange_rates(self.settings.rates)
            self.exchanger.save_rates(self.settings.rates)
            
        print(f"[INFO]: Exchange rates: {self.settings.rates}")
        
    def get_researcher(self, profession_name: str) -> tuple:
        """Создает экземпляры Analyzer и DataCollector для указанной профессии"""
        # Обновляем настройки для текущей профессии
        self.settings.options["text"] = profession_name
        
        # Создаем объекты для анализа
        collector = DataCollector(self.settings.rates)
        
        # Явно передаем имя профессии
        analyzer = Analyzer(self.settings.save_result, profession_name)
        print(f"[INFO]: Создан анализатор для профессии: {profession_name}")
        
        return analyzer, collector
        
    def process_all(self):
        """Обработка всех профессий из списка"""
        print(f"[INFO]: Will process {len(self.professions)} professions:")
        for i, prof in enumerate(self.professions):
            print(f"  {i+1}. {prof}")
        
        # Обрабатываем каждую профессию
        for i, profession in enumerate(self.professions):
            print("\n" + "=" * 80)
            print(f"[INFO]: Processing profession {i+1}/{len(self.professions)}: {profession}")
            print("=" * 80)
            
            try:
                # Создаем анализатор и коллектор для профессии
                analyzer, collector = self.get_researcher(profession)
                
                # Создаем директорию для сохранения результатов
                output_dir = analyzer.get_output_directory()
                os.makedirs(output_dir, exist_ok=True)
                print(f"[INFO]: Created directory for profession: {output_dir}")
                
                # Собираем данные
                vacancies = collector.collect_vacancies(
                    query={"text": profession, "area": self.settings.options.get("area", 113), 
                           "per_page": self.settings.options.get("per_page", 100)},
                    refresh=self.settings.refresh,
                    num_workers=self.settings.num_workers
                )
                
                # Извлекаем статистику
                stats = vacancies.pop("stats", None)
                
                # Подготавливаем DataFrame
                df = analyzer.prepare_df(vacancies)
                
                # Анализируем данные
                print("\n[INFO]: Analyzing dataframe...")
                analyzer.analyze_df(df)
                
                # Сохраняем анализ
                output_filename = os.path.join(output_dir, "salary_analysis.csv")
                analyzer.analyze_and_save_results(df, output_filename=output_filename, stats=stats)
                
            except Exception as e:
                print(f"[ERROR]: Failed to process profession '{profession}': {str(e)}")
                traceback.print_exc()
                continue
                
        print("\n[INFO]: All professions processing completed!")
            
if __name__ == "__main__":
    processor = MultiProcessor()  # Теперь будет считывать профессии из professions.txt
    processor.process_all()