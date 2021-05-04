import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

for well in ['5351', '5599', '7078', '7289', '7405f']:
    train_data_path = f'input_data/{well}_target.csv'
    train_data = pd.read_csv(train_data_path)

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=100,
                                    max_window_size=30,
                                    make_future_prediction=True))

    data = InputData(idx=np.arange(len(train_data)),
                     target=train_data.target,
                     features=None,
                     task=task,
                     data_type=DataTypesEnum.ts)

    model = Fedot(problem='ts_forecasting',
                  task_params=TsForecastingParams(forecast_length=100,
                                                  max_window_size=30,
                                                  make_future_prediction=True))

    chain = model.fit(features=data, predefined_model='rfr')

    ts_forecast = model.forecast(pre_history=data, forecast_length=100)
    from numpy import savetxt
    ml = ts_forecast
    savetxt(f'final/ml{well}.csv', ml, delimiter=',')