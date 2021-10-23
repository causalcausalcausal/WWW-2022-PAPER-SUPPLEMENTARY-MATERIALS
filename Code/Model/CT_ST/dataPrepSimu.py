import pandas as pd

synthetic_data_dir = '../../../Data/Synthetic_data/'
output_data_dir = './data_prep/'

for uw in [5, 10, 15, 20]:
    for type in ['Training', 'Test']:
        fp = '%dWeight_SimulationDataAll%s.csv' % (uw, type)
        df = pd.read_csv(synthetic_data_dir + fp)
        df_control = df[df.A == 0]
        for treatment in [1, 2, 3]:
            print('process uw = %d, type = %s, treatment = %d' % (uw, type, treatment))
            df_treatment = df[df.A == treatment].copy()
            df_treatment.A = 1
            df_out = pd.concat([df_treatment, df_control])
            out_fp = "A%d_%dWeight_SimulationData%s.csv" % (treatment, uw, type)
            df_out.to_csv(output_data_dir + out_fp, index=False)