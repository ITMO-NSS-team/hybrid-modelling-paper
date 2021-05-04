from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

for well in ['5351', '5599', '7078', '7289', '7405f']:
    prod1 = pd.read_csv(f'final/prod{well}.csv').values
    crm1 = pd.read_csv(f'final/crm{well}.csv').values
    ml_crm1 = pd.read_csv(f'final/ml_crm{well}.csv').values
    ml1 = pd.read_csv(f'final/ml{well}.csv').values
    hybrid1 = pd.read_csv(f'final/ml_hybrid{well}.csv').values

    prod = prod1[0:50]
    crm = crm1[0:50]
    ml_crm = ml_crm1[0:50]
    ml = ml1[0:50]
    hybrid = hybrid1[0:50]

    print(f"Metrics for {well}")

    print('MAE for CRM:', mean_absolute_error(prod, crm))
    print('MAE for ML:', mean_absolute_error(prod, ml))
    print('MAE for ML_CRM:', mean_absolute_error(prod, ml_crm))
    print('MAE for Hybrid_ML:', mean_absolute_error(prod, hybrid))

    print('RMSE for CRM:', mean_squared_error(prod, crm, squared=False))
    print('RMSE for ML:', mean_squared_error(prod, ml, squared=False))
    print('RMSE for ML_CRM:', mean_squared_error(prod, ml_crm, squared=False))
    print('RMSE for Hybrid_ML:', mean_squared_error(prod, hybrid, squared=False))

    print('MAPE for CRM:', mean_absolute_percentage_error(prod, crm))
    print('MAPE for ML:', mean_absolute_percentage_error(prod, ml))
    print('MAPE for ML_CRM:', mean_absolute_percentage_error(prod, ml_crm))
    print('MAPE for Hybrid_ML:', mean_absolute_percentage_error(prod, hybrid))

    if max(prod) > max(crm):
        high_border = max(prod)
    else:
        high_border = max(crm)

    f, axs = plt.subplots(figsize=(15, 7))
    ax = plt.gca()
    plt.axis([0, len(prod), 0, high_border+50])
    ax.set_autoscale_on(False)
    ax.set_title(f'Debit of {well}')

    plt.plot(prod, label="Observed")
    plt.plot(crm, label="CRM")
    plt.plot(ml, label="ML")
    plt.plot(hybrid, label="Hybrid ML")

    plt.legend(labels=['Observed', 'CRM', 'ML', 'Hybrid ML'])

    ax.set_xlabel('Day of debit (day)')
    ax.set_ylabel('Oil debit (m³/day)')
    plt.savefig(f'{well}YSC.eps', format='eps')
    plt.savefig(f'{well}YSC.png')