import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, path, dpi):
        self._path = path
        self._dpi = dpi

    def save_plot_and_data(self, data, filename, xlabel, ylabel):
        min_val, max_val = min(data), max(data)
        plt.rcParams.update({'font.size': 24})  
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))

        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        plot_path = os.path.join(self._path, f'plot_{filename}.png')
        fig.savefig(plot_path, dpi=self._dpi)
        plt.close("all")

        data_file_path = os.path.join(self._path, f'plot_{filename}_data.txt')
        with open(data_file_path, "w") as file:
            for value in data:
                file.write(f"{value}\n")
