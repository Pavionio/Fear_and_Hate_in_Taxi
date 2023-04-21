# Fear_and_Hate_in_Taxi
Основной файл - [catch_anomalies_function.py](catch_anomalies_function.py)  
В нём находится функция - Catch_Anomalies:
          Функция для выявления аномалий в данных поездок.

          data - pd.DataFrame - датасет с данными    
          search_geo_anomalies -  добавляет колонку, где отмечает поездки, которые находятся вне Нью-Йорка и рисует большую карту Нью-Йорка  (не рекоммендуется использовать на больших датасетах)
          - необходимы колонки:
          pickup_lagitude
          pickup_longitude
          dropoff_longitude
          dropoff_lagitude

          show_small_geo_map - рисует координаты поездок на карте Америки (не рекоммендуется использовать на больших датасетах)
          - необходимы колонки:
          pickup_lagitude
          pickup_longitude
          dropoff_longitude
          dropoff_lagitude

          search_count_anomalies_by_iso - возвращает аномалии в количестве поездок по дням используя isolation forest
          - необходимы колонки:
           pickup_datetime

          search_count_anomalies_by_prophet - возвращает аномалии в количестве поездок по дням используя prohpet (num_of_periods - количество дней которые prophet предскажет на графике)
          - необходимы колонки:
           pickup_datetime

          search_nlo_anomalies -  добавляет колонку аномалий, которые находит LocalOutlierFactor и рисует график (не рекоммендуется использовать на больших датасетах)
          - необходимы колонки:
           trip_duration
           distance

          search_parameter_anomalies - ищет логически невозможные значения
          - необходима хотя бы одна колонка из следующих:
          trip_duration
          distance
          passenger_count


Разведочный анализ - explore.ipynb, clean_data.ipynb, time_series_res.ipynb
Использование инструментов - clean_data.ipynb, isoforest_demo.ipynb, remove_anomalies.ipynb, anomaly_function.ipynb