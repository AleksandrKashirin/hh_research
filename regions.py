import requests
from typing import List, Dict

def get_areas() -> List[Dict]:
    """Получает список всех регионов"""
    response = requests.get('https://api.hh.ru/areas')
    return response.json()

def get_russian_cities(areas: List[Dict]) -> List[Dict]:
    """Получает все города России из дерева регионов"""
    cities = []
    
    def collect_cities(area: Dict, parent_name: str = ""):
        # Если это город (нет подобластей), добавляем его
        if not area.get('areas'):
            cities.append({
                'id': area['id'],
                'name': area['name'],
                'parent': parent_name
            })
        # Если есть подобласти, рекурсивно обрабатываем их
        else:
            for subarea in area['areas']:
                collect_cities(subarea, area['name'])
    
    # Находим Россию и обрабатываем её регионы
    for country in areas:
        if country['id'] == '113':  # id России
            for region in country['areas']:
                collect_cities(region)
            break
    
    return cities

def main():
    # Получаем все регионы
    print("Загрузка данных из API...")
    areas = get_areas()
    
    # Получаем все города России
    cities = get_russian_cities(areas)
    
    # Сортируем города по алфавиту
    cities.sort(key=lambda x: x['name'])
    
    # Выводим результаты
    print(f"\nНайдено городов: {len(cities)}")
    print("\nПример вывода (первые 20 городов):")
    print("-" * 70)
    print(f"{'ID':<8} | {'Город':<30} | {'Регион':<30}")
    print("-" * 70)
    
    for city in cities[:20]:
        print(f"{city['id']:<8} | {city['name']:<30} | {city['parent']:<30}")
    
    # Сохраняем полный список в файл
    with open('russian_cities.txt', 'w', encoding='utf-8') as f:
        f.write(f"{'ID':<8} | {'Город':<30} | {'Регион':<30}\n")
        f.write("-" * 70 + "\n")
        for city in cities:
            f.write(f"{city['id']:<8} | {city['name']:<30} | {city['parent']:<30}\n")
    
    print(f"\nПолный список сохранен в файл 'russian_cities.txt'")
    
    # Показываем как использовать area_id для поиска по всей России
    print("\nДля поиска по всей России используйте:")
    print("area=113 в параметрах запроса")
    print("\nПример URL для поиска:")
    print("https://api.hh.ru/vacancies?area=113&text=Кавист")

if __name__ == "__main__":
    main()