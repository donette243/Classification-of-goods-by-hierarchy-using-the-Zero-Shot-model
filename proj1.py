import pandas as pd
import re
from transformers import pipeline
from tqdm import tqdm  
import plotly.express as px


df = pd.read_csv('C:/Users/Donete/Documents/Projets/Drone/projet/produits.txt', encoding='windows-1251', sep='\t')

def очистить_текст(текст):
    текст = str(текст).lower()
    текст = re.sub(r'[^\w\s]', '', текст)
    текст = re.sub(r'\s+', ' ', текст).strip()
    return текст

df['очищенное_описание'] = df['description'].apply(очистить_текст)

# Определение 20 иерархических категорий (упрощённые UNSPSC/ECLASS)
метки = [
    "Электроника > Компьютеры > Ноутбуки",
    "Электроника > Аудио > Наушники",
    "Электроника > Видео > Мониторы",
    "Электроника > Переферия > Принтеры",
    "Бытовая техника > Кухня > Микроволновки",
    "Бытовая техника > Уборка > Пылесосы",
    "Бытовая техника > Уход за собой > Фены",
    "Мебель > Столы > Кофейные столики",
    "Мебель > Освещение > Настольные лампы",
    "Мода > Одежда > Куртки",
    "Мода > Обувь > Кроссовки",
    "Мода > Аксессуары > Сумки",
    "Дом > Инструменты > Наборы инструментов",
    "Дом > Освещение > Гирлянды",
    "Спорт > Йога > Коврики",
    "Спорт > Тренажёры > Тренажёры для пресса",
    "Дети > Игрушки > Машинки",
    "Дети > Игрушки > Конструкторы",
    "Авто > Аксессуары > Зарядные устройства",
    "Офис > Оборудование > Лампы"
]

# Создание zero-shot pipeline
локальная_модель = "C:/Users/Donete/Documents/Projets/Drone/Models/"
классификатор = pipeline("zero-shot-classification", model=локальная_модель)

# Применение классификации к каждой строке
предсказания = []
for текст in tqdm(df['очищенное_описание'], desc="Классификация"):
    результат = классификатор(текст, метки, multi_label=False)
    предсказания.append(результат['labels'][0])

# Добавление предсказаний в DataFrame
df['предсказанная_категория'] = предсказания

# Отображение нескольких результатов
print(df[['description', 'предсказанная_категория']].head())

# Сохранение результатов в новый CSV файл
df.to_csv("C:/Users/Donete/Documents/Projets/Drone/projet/produits_classes.csv", index=False, encoding='utf-8-sig')

# Повторная загрузка файла (с пропуском ошибочных строк)
путь_к_файлу = "C:/Users/Donete/Documents/Projets/Drone/projet/produits_classes.csv"
df = pd.read_csv(
    путь_к_файлу,
    encoding='utf-8',
    sep=',',
    on_bad_lines='skip',
    engine='python'
)


# Проверка наличия нужной колонки
if 'предсказанная_категория' not in df.columns:
    print(" Колонка 'предсказанная_категория' не найдена. Доступные колонки:", df.columns.tolist())
    exit()

# Разделение категорий на уровни
df[['семейство', 'категория', 'класс']] = df['предсказанная_категория'].str.split(' > ', n=2, expand=True)

# Построение sunburst-графика
fig = px.sunburst(
    df,
    path=['семейство', 'категория', 'класс'],
    title='Распределение товаров по иерархии категорий',
    color='семейство'
)
fig.show()

# Сохранение графика в PNG
fig.write_image("C:/Users/Donete/Documents/Projets/Drone/projet/sunburst_categories.png")

