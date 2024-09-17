# OpenEyesClassificator
## Запуск кода классификации
0. Создать и активировать окружение
 ```
python3 -m venv envOpenEyesClassificator
source envOpenEyesClassificator/bin/activate
 ```
1. Склонировать репозиторий и перейти в его папку
```
git clone https://github.com/ML-MountainLover/OpenEyesClassificator.git
cd OpenEyesClassificator
```
2. Установить зависимости
```
pip install -r requirements.txt
```
3. Создать файл окружения, добавить туда ID файла с чекпоинтом на гугл-диске
```
touch .env && echo DRIVE_ID=1E2tscC_VKhJKLciRwJEPRFbacfoZRanU > .env
```
4. Запустить код на тестовом примере
```
python3 classificator.py
```
## Запуск кода тренировки
0. Если пункты 0-2 из запуска кода классификации еще не проделаны - проделать их
1. Загрузить данные для обучения
```
gdown 122BgFHJG8Kgn1E_bkT1Lu8I1Cf10glVn
unzip /content/EyesDataset.zip
```
2. Запустить код на тестовом примере
```
python3 train.py
```
Данный код запускает 2 эпохи тренировки на финальном датасете модели kaggle_model (См. отчет пункт 3.3)

## Отчет
  [Ссылка на отчет](report.pdf)