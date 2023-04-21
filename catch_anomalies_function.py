import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import geopandas as gpd
import math
from shapely.geometry import Point
import shapely
import json
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error

def geo_anomalies(data: pd.DataFrame) -> None:
    df_for_pickup = pd.read_csv('/content/gdrive/MyDrive/intensive/Project/taxi_zones.csv')
    df_for_dropoff = pd.read_csv(
        '/content/gdrive/MyDrive/intensive/Project/taxi_zones.csv')  # update with the great help of Pavel Fadeev

    polygon_for_pickup = gpd.GeoDataFrame(geometry=shapely.wkt.loads(df_for_pickup.the_geom))
    # polygon_for_dropoff = gpd.GeoDataFrame(geometry=shapely.wkt.loads(df_for_dropoff.the_geom))

    ls = []

    for _, row in data.iterrows():
        ls.append(polygon_for_pickup.contains(
            Point(row['pickup_longitude'], row['pickup_latitude'])).any() and polygon_for_pickup.contains(
            Point(row['dropoff_longitude'], row['dropoff_latitude'])).any())

    pickup_longitude = data[['id', 'pickup_longitude']].rename(columns={'pickup_longitude': 'longitude'})
    pickup_longitude['type'] = 'pickup'
    dropoff_longitude = data[['id', 'dropoff_longitude']].rename(columns={'dropoff_longitude': 'longitude'})
    dropoff_longitude['type'] = 'dropoff'

    longitudes = pd.concat((pickup_longitude, dropoff_longitude))

    pickup_latitude = data[['id', 'pickup_latitude']].rename(columns={'pickup_latitude': 'latitude'})
    pickup_latitude['type'] = 'pickup'
    dropoff_latitude = data[['id', 'dropoff_latitude']].rename(columns={'dropoff_latitude': 'latitude'})
    dropoff_latitude['type'] = 'dropoff'

    latitudes = pd.concat((pickup_latitude, dropoff_latitude))

    coords = pd.merge(longitudes, latitudes, on=['id', 'type'])

    with open('/content/gdrive/MyDrive/intensive/Project/NYC Taxi Zones.geojson') as f:
        taxi_zones = json.load(f)

    fig = px.scatter_mapbox(coords, lat='latitude', lon='longitude', hover_name='type')
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(mapbox_bounds={"west": -180, "east": -50, "south": 20, "north": 90})
    fig.update_traces(cluster=dict(enabled=True))
    fig.update_layout(
        mapbox={
            'style': "open-street-map",
            'center': {'lon': -73.9662, 'lat': 40.7834},
            'zoom': 9, 'layers': [{
                'source': taxi_zones,
                'type': "fill", 'below': "traces", 'color': "gray", 'name': 'Область работы такси',
                'opacity': 0.5}]},
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    fig.show()

    data['geo_anomalies'] = ls


def show_geo_map(data: pd.DataFrame) -> None:
    pickup = data[['pickup_latitude', 'pickup_longitude']].rename(
        columns={'pickup_longitude': 'longitude', 'pickup_latitude': 'latitude'})
    dropoff = data[['pickup_latitude', 'pickup_longitude']].rename(
        columns={'pickup_longitude': 'longitude', 'pickup_latitude': 'latitude'})

    coords = pd.concat((pickup, dropoff))
    geo_coords = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords.longitude, coords.latitude))

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world[world.continent == 'North America'].plot(color='white', edgecolor='black')
    geo_coords.plot(ax=ax, color='purple')  # Расположение точек начала и конца поездок
    plt.show()


def iso_forest_trip_count_anomalies(data: pd.DataFrame) -> None:
    data = data.sort_values(by=['pickup_datetime'])
    cols = data.columns
    data['ds'] = pd.to_datetime(data['pickup_datetime'].str.slice(stop=10))
    duration = data.groupby('ds', as_index=False).count()
    duration.rename(columns={cols[2]: "y"}, inplace=True)
    duration = duration[['ds', 'y']]
    model = IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.05), max_features=1.0)

    model.fit(duration[['y']])

    score_anomalies = model.decision_function(duration[['y']])
    # print('Isolation_Forest score is ', np.mean([-1*s + 0.5 for s in score_anomalies])

    duration['day_anomaly'] = model.predict(duration[['y']])

    an = duration.loc[duration['day_anomaly'] == -1]
    an.sort_values(by=['y'])

    fig = plt.figure(figsize=(15, 10))
    plt.plot(duration.ds, duration['y'], c='#264FA3', linewidth=3.0)
    plt.scatter(an.ds, an['y'], c='red', s=9, linewidth=5.0)
    plt.show()

    return duration


def trip_nlo_anomalies(data: pd.DataFrame) -> None:
    LOF = LocalOutlierFactor(n_neighbors=50, contamination='auto')
    x = data[['trip_duration', 'distance']].values
    y_pred = LOF.fit_predict(x)

    plt.figure(figsize=(12, 12))

    in_mask = [True if i == 1 else False for i in y_pred]
    out_mask = [True if i == -1 else False for i in y_pred]

    plt.title("Local Outlier Factor (LOF)")

    a = plt.scatter(x[in_mask, 0], x[in_mask, 1], c='blue', edgecolor='k', s=30)

    b = plt.scatter(x[out_mask, 0], x[out_mask, 1], c='red', edgecolor='k', s=30)
    plt.axis('tight')
    plt.xlabel('trip_duration');
    plt.ylabel('distance');
    plt.show()

    data['nlo_anomaly'] = y_pred


def trip_parameter_anomalies(data: pd.DataFrame) -> None:
    passenger_count_m = False
    passenger_count_l = False
    distance_anomaly = False
    duration_anomaly = False

    if 'distance' in data.columns:
        distance_anomaly = data.distance == 0

    if 'trip_duration' in data.columns:
        duration_anomaly = data.trip_duration == 0

    if 'trip_duration' in data.columns:
        passenger_count_m = data.passenger_count > 7
        passenger_count_l = data.passenger_count == 0

    data['parameter_anomaly'] = 0
    data.loc[
        (passenger_count_m) | (passenger_count_l) | (distance_anomaly) | (duration_anomaly), 'parameter_anomaly'] = -1


def get_distance(p1, p2):
    # p1 и p2 - это кортежи из двух элементов - координаты точек
    radius = 6373.0

    lon1 = math.radians(p1[0])
    lat1 = math.radians(p1[1])
    lon2 = math.radians(p2[0])
    lat2 = math.radians(p2[1])

    d_lon = lon2 - lon1
    d_lat = lat2 - lat1

    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(a ** 0.5, (1 - a) ** 0.5)

    distance = radius * c
    return distance


def prophet_trip_count_anomalies(data: pd.DataFrame, num_of_periods: int) -> None:
    data = data.sort_values(by=['pickup_datetime'])
    cols = data.columns
    data['ds'] = pd.to_datetime(data['pickup_datetime'].str.slice(stop=10))

    duration = data.groupby('ds', as_index=False).count()
    duration.rename(columns={cols[1]: "y"}, inplace=True)
    duration = duration[['ds', 'y']]

    m = Prophet()
    m.fit(duration)

    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    m.plot(forecast)
    plt.show()

    m.plot_components(forecast)
    plt.show()

    performance = pd.merge(duration, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

    performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
    print(f'\n \n The MAE for the model is {performance_MAE}')

    performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
    print(f'The MAPE for the model is {performance_MAPE} \n \n')

    performance['anomaly'] = performance.apply(
        lambda rows: 1 if ((rows.y <= rows.yhat_lower) | (rows.y >= rows.yhat_upper)) else 0, axis=1)

    anomalies = performance[performance['anomaly'] == 1].sort_values(by='ds')

    fig = plt.figure(figsize=(15, 10))

    sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly', linewidth=0.001).set(title='Количество поездок',
                                                                                         ylabel='Количество поездок',
                                                                                         xlabel='Дата')
    sns.lineplot(x='ds', y='yhat', data=performance, color='black', linewidth=2)

    return performance


def Catch_Anomalies(data: pd.DataFrame, search_geo_anomalies=False, show_small_geo_map=False,
                    search_count_anomalies_by_iso=True, \
                    search_count_anomalies_by_prophet=True, num_of_periods=30, search_nlo_anomalies=False,
                    search_parameter_anomalies=True):
    """
          Функция для выявления аномалий в данных поездок.

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



    """

    day_anomalies_iso = None
    day_anomalies_prophet = None

    if 'pickup_longitude' in data.columns and search_geo_anomalies:
        geo_anomalies(data)

    if 'pickup_longitude' in data.columns and show_small_geo_map:
        show_geo_map(data)

    if 'pickup_datetime' in data.columns and search_count_anomalies_by_iso:
        day_anomalies_iso = iso_forest_trip_count_anomalies(data)

    if 'pickup_datetime' in data.columns and search_count_anomalies_by_prophet:
        day_anomalies_prophet = prophet_trip_count_anomalies(data, num_of_periods)

    if 'distance' and 'trip_duration' in data.columns and search_nlo_anomalies:
        trip_nlo_anomalies(data)

    if search_parameter_anomalies:
        trip_parameter_anomalies(data)

    return day_anomalies_iso, day_anomalies_prophet
