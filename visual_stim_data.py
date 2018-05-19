import numpy as np
import random
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class VisualStimData:
    """
    Data and methods for the visual stimulus ePhys experiment.
    The data table itself is held in self.data, an `xarray` object.
    Inputs:
        data: xr.DataArray or xr.Dataset
        ...
    Methods:
         ...
    """

    def __init__(self, data, stimulus_start=1000, stimulus_duration=100):
        self.data = data
        self.stimulus_start = stimulus_start
        self.stimulus_duration = stimulus_duration
        self.experimenter_names = {data.attrs['experimenter_name'] for name, data in self.data.data_vars.items()}
        self.stats_to_cal = ('mean', 'median', 'std')

    def plot_electrode(self, rep_number: int, rat_id: int, elec_number: tuple = (0,)):
        """
        Plots the voltage of the electrodes in "elec_number" for the rat "rat_id" in the repetition
        "rep_number". Shows a single figure with subplots.
        """
        fig, axs = plt.subplots(len(elec_number), 1)
        for plot, electrode in enumerate(elec_number):
            x_data = self.data.coords['time'].values
            y_data = self.data[rat_id].sel(electrode=electrode, repetition=rep_number)
            axs[plot].plot(x_data, y_data, linewidth=0.1)
            axs[plot].set_title(f'Electrode #{electrode}')
        plt.show()

    def experimenter_bias(self):
        """ Shows the statistics of the average recording across all experimenters """
        all_stats = pd.DataFrame(columns=['experimenter', 'statistic', 'value'])
        for experimenter in self.experimenter_names:
            individual_stats = self.calculate_individual_stats(experimenter)
            individual_stats = individual_stats.melt(id_vars='experimenter', var_name='statistic')
            all_stats = all_stats.append(individual_stats)
        sns.factorplot(x='experimenter', y='value', hue='statistic', data=all_stats, kind='bar')
        plt.show()

    def calculate_individual_stats(self, experimenter):
        """ Calculates each experimenters statistics individually """
        individual_data = self.data.filter_by_attrs(experimenter_name=experimenter)
        mean_data = individual_data.mean(dim='repetition').to_dataframe().groupby('time').mean().mean(axis='columns')
        individual_stats = pd.DataFrame(columns=['experimenter']+list(self.stats_to_cal))
        individual_stats.loc[0, 'experimenter'] = experimenter
        for stat_to_cal in self.stats_to_cal:
            value = getattr(np, stat_to_cal)(mean_data)
            individual_stats.loc[0, stat_to_cal] = value
        return individual_stats


def mock_stim_data() -> VisualStimData:
    """ Creates a new VisualStimData instance with mock data """
    room_temp_range = (0, 40)
    room_humidity_range = (0, 100)
    experimenter_names = ('Ayam', 'Hagai', 'Pablo')
    rat_gender_options = ('Male', 'Female')
    electrodes_num = 10
    sample_rate = 10000
    repetitions_num = 4
    rats_num = 10
    trial_duration = 2000
    dims = ('time', 'repetition', 'electrode')
    coordinates = {'time': np.linspace(0, trial_duration, sample_rate),
                   'repetition': np.arange(repetitions_num),
                   'electrode': np.arange(electrodes_num)}
    data = xr.Dataset()
    for i in range(rats_num):
        temp_data = {'rat_id': i,
                     'room_humidity': np.random.uniform(low=room_humidity_range[0], high=room_humidity_range[1]),
                     'room_temp': np.random.uniform(low=room_temp_range[0], high=room_temp_range[1]),
                     'experimenter_name': random.choice(experimenter_names),
                     'rat_gender': random.choice(rat_gender_options)}
        final_data = xr.DataArray(np.random.random((sample_rate, repetitions_num, electrodes_num)),
                                dims=dims,
                                coords=coordinates, attrs=temp_data)
        data[i] = final_data
    return data


if __name__ == '__main__':
    stim_data = mock_stim_data()
    stim_data = VisualStimData(stim_data)
    stim_data.plot_electrode(rep_number=3, rat_id=2, elec_number=(0, 1, 2, 5))  # add necessary vars
    stim_data.experimenter_bias()