import os
import random
import yaml

import pandas as pd
from omegaconf import DictConfig

class FilterData():
    def __init__(self, config):
        self.config = config

    @staticmethod
    def filter_out_zero_values(self, df, *column_index_to_save):
        for col_index in column_index_to_save:
            df = (df.iloc[:, col_index] != 0)
        return df
    
    @staticmethod
    def replace_labels_with_legend(self, df, **column_index_to_legend):
        # label -> corresponding legend
        for col_index, legend in column_index_to_legend.items():
                df.iloc[:, col_index] = df.iloc[:, col_index].astype(str).replace(legend)

        return df
        
    @staticmethod
    def select_columns(self, df, *column_index_to_save):
        # Selecting only the columns of interest for locomotion and ML_Both_Arms
        df = df.iloc[:, column_index_to_save]
        return df
    
    @staticmethod
    def extract_moments_of_change(df):
        # Extract moments where values are changing
        df = df[df.ne(df.shift()).any(axis=1)]
        return df

    @staticmethod
    def add_relative_time_column(df, sampling_freq):
        df['relative_time(s)'] = (df.index / sampling_freq).astype(int)
        return df

    @staticmethod
    def rename_columns(df):
        df.rename(columns={243: 'locomotion_label_0', 249: 'gesture_label_0'}, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
    
class MislabelingProcessor():
    def __init__(self, config):
        self.config = config
        self.indices_to_mislabel = None

    @staticmethod
    def wild_mislabeling(self, df, mislabeling_mode, target_labels, mislabeling_rate, num_label_duplication, num_label_mislabeling):
        for label in target_labels:
            num_row_mislabeling = int(df[df[label]] * mislabeling_rate)
            
        
    @staticmethod
    def random_mislabeling(self, df, mislabeling_mode, target_labels, mislabeling_rate, num_label_duplication, num_label_mislabeling):
        num_row_mislabeling = int(len(df) * mislabeling_rate)
        self.indices_to_mislabel = random.sample(range(len(df)), num_row_mislabeling)

        for row_index in self.indices_to_mislabel:
            mislabeling_column_index = random.sample(range(num_label_duplication), num_label_mislabeling)
            for col_index in mislabeling_column_index:
                for label in target_labels:
                    if mislabeling_mode == "wild":
                        df.loc[row_index, f"{label}_label_{col_index}"] = self.generate_flip_label(df, row_index, label, col_index)
                    elif mislabeling_mode == "fused":
                        df.loc[row_index, f"{label}_label_{col_index}"] = 'Swim'
        return df



def filter_data(
    config: DictConfig,
):
    match config.dataset_name:
        case "opportunity":
            with open(config.metadata_path, "r") as file:
                metadata = yaml.safe_load(file)

            column_index_to_save = [
                metadata.locomotion_column_index,
                metadata.ML_Both_Arms_column_index
            ]
            column_index_to_legend = {
                metadata.locomotion_column_index: metadata.locomotion_label_legend,
                metadata.ML_Both_Arms_column_index: metadata.ML_Both_Arms_label_legend
            }
            sensor_frequency = metadata.sampling_frequency_hz

            df = pd.read_csv(
                os.path.join(metadata.dataset_directory, metadata.adl_name),
                sep=' ',
                header=None
            )
            
            df = FilterData.filter_out_zero_values(df, column_index_to_save)
            df = FilterData.replace_labels_with_legend(df, column_index_to_legend)

            df = FilterData.select_columns(df, column_index_to_save)
            df = FilterData.extract_moments_of_change(df)
            df = FilterData.add_relative_time_column(df, sensor_frequency)
            df = FilterData.rename_columns(df)
            return df
    
        case "your-dataset-name":
            # Add preprocess code here.
            pass
        case default:
            raise ValueError(f"Dataset {config.dataset_name} not recognized.")

def apply_misclassification(
    df: pd.DataFrame,
    config: DictConfig,
):
    match config.dataset_name:
        case "opportunity":
            pass
        case "your-dataset-name":
            # Add preprocess code here.
            pass
        case default:
            raise ValueError(f"Dataset {config.dataset_name} not recognized.")
    