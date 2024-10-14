import yaml
import os

from omegaconf import DictConfig
import pandas as pd

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

def filter_data(
    df: pd.DataFrame,
    config: DictConfig,
    metadata: DictConfig
):
    match config.dataset_name:
        case "opportunity":
            column_index_to_save = [
                metadata.locomotion_column_index,
                metadata.ML_Both_Arms_column_index
            ]
            column_index_to_legend = {
                metadata.locomotion_column_index: metadata.locomotion_label_legend,
                metadata.ML_Both_Arms_column_index: metadata.ML_Both_Arms_label_legend
            }
            sensor_frequency = metadata.sampling_frequency_hz
            
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
        

def load_datasets(
    config: DictConfig,
):
    with open(config.metadata_path, "r") as file:
        metadata = yaml.safe_load(file)

    df = pd.read_csv(
        os.path.join(metadata.dataset_directory, metadata.adl_name),
        sep=' ',
        header=None
    )
            
    filtered_dataset = filter_data(df, config, metadata)

    return filtered_dataset
