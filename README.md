# Страх и ненависть в Нью-йоркском такси
Основной файл - [catch_anomalies_function.py](catch_anomalies_function.py)  
В нём находится функция - Catch_Anomalies:
          Функция для выявления аномалий в данных поездок.

                    Функция для выявления аномалий в данных поездок.
          data - pd.DataFrame - датасет с данными
          search_geo_anomalies -  добавляет колонку, где отмечает поездки, которые находятся вне Нью-Йорка и рисует большую карту Нью-Йорка  (не рекомендуется использовать на больших датасетах)
          По умолчанию -  False
          - необходимы колонки:
          pickup_lagitude
          pickup_longitude
          dropoff_longitude
          dropoff_lagitude

          show_small_geo_map - рисует координаты поездок на карте Америки (не рекомендуется использовать на больших датасетах)
          По умолчанию -  False
          - необходимы колонки:
          pickup_lagitude
          pickup_longitude
          dropoff_longitude
          dropoff_lagitude

          search_count_anomalies_by_iso - возвращает аномалии в количестве поездок по дням используя isolation forest
          По умолчанию -  True
          - необходимы колонки:
           pickup_datetime

          search_count_anomalies_by_prophet - возвращает аномалии в количестве поездок по дням используя prophet
          По умолчанию -  True
          - необходимы колонки:
           pickup_datetime

          search_nlo_anomalies -  добавляет колонку аномалий, которые находит LocalOutlierFactor и рисует график (не рекомендуется использовать на больших датасетах)
          По умолчанию -  False
          - необходимы колонки:
           trip_duration
           distance

          search_parameter_anomalies - ищет логически невозможные значения
          - необходима хотя бы одна колонка из следующих:
          По умолчанию -  True
          trip_duration
          distance
          passenger_count


Разведочный анализ - explore.ipynb, clean_data.ipynb, time_series_res.ipynb  
Использование инструментов - clean_data.ipynb, isoforest_demo.ipynb, remove_anomalies.ipynb, anomaly_function.ipynb