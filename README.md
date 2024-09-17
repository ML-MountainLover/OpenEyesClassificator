# OpenEyesClassificator
## Запуск кода
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
## Отчет
  [Ссылка на отчет](report.pdf)