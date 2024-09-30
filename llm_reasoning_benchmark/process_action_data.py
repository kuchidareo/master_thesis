import os
import random

import pandas as pd

from config import UCI_dataset_config

class UCIDataHandler():
    dataset_directory = "OpportunityUCIDataset/dataset"
    filtered_csv_directory = "filtered_action_csv"
    config = UCI_dataset_config

    def __init__(self, dataset_name, num_sensor_labels, poisoning_conf) -> None:
        self.dataset_name = dataset_name
        self.num_sensor_labels = num_sensor_labels
        self.poisoning_conf = poisoning_conf

    def load_df(self):
        return pd.read_csv(os.path.join(self.dataset_directory, self.dataset_name), sep=' ', header=None)
    
    def filter(self, df):
        locomotion_col_idx = self.config["locomotion_column_index"]
        gesture_col_idx = self.config["ML_Both_Arms_column_index"]

        # Filter out rows with 0 in locomotion and ML_Both_Arms columns
        locomotion_label = df.iloc[:, locomotion_col_idx]
        gesture_label = df.iloc[:, gesture_col_idx]    

        valid_rows = (locomotion_label != 0) | (gesture_label != 0)

        df =  df[valid_rows]

        # label -> corresponding legend
        locomotion_label_legend = self.config["locomotion_label_legend"]
        ml_both_arms_label_legend = self.config["ML_Both_Arms_label_legend"]
        
        df.iloc[:, locomotion_col_idx] = df.iloc[:, locomotion_col_idx].astype(str).replace(locomotion_label_legend)
        df.iloc[:, gesture_col_idx] = df.iloc[:, gesture_col_idx].astype(str).replace(ml_both_arms_label_legend)

        # Selecting only the columns of interest for locomotion and ML_Both_Arms
        df = df.iloc[:, [locomotion_col_idx, gesture_col_idx]]

        # Extract moments where values are changing
        df = df[df.ne(df.shift()).any(axis=1)]

        sampling_freq = self.config["sampling_frequency_hz"]
        df['relative_time(s)'] = (df.index / sampling_freq).astype(int)

        df.rename(columns={243: 'locomotion_label_0', 249: 'gesture_label_0'}, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def poisoning(self, df):
        attack_labels = self.poisoning_conf["attack_labels"]

        poisoned_df = df.copy()
        poisoned_df = self.duplicate_attack_columns(poisoned_df, attack_labels)

        if self.poisoning_conf["label_mode"] == "swim":
            if self.poisoning_conf["position"]["in_column"] == "random":
                num_rows_to_poison = int(len(poisoned_df) * self.poisoning_conf["position"]["rate"])
                indices_to_poison = random.sample(range(len(poisoned_df)), num_rows_to_poison)

                for i in indices_to_poison:
                    for j in range(self.poisoning_conf["position"]["num_of_column"]):
                        for label in attack_labels:
                            poisoned_df.loc[i, f"{label}_label_{j}"] = 'Swim'


        column_order = sorted(poisoned_df.columns)
        poisoned_df = poisoned_df[column_order]

        return poisoned_df

    def duplicate_attack_columns(self, df, attack_labels):
        for label in attack_labels:
            for i in range(1, self.num_sensor_labels):
                df[f"{label}_label_{i}"] = df[f"{label}_label_0"].copy()
        return df


    def export_csv(self, df):
        attack_labels = str(self.poisoning_conf["attack_labels"])
        label_mode = self.poisoning_conf["label_mode"]
        in_column = self.poisoning_conf["position"]["in_column"]
        num_of_column = self.poisoning_conf["position"]["num_of_column"]
        rate = self.poisoning_conf["position"]["rate"]

        csv_name = f"{dataset_name}_{self.num_sensor_labels}_{attack_labels}_{label_mode}_{in_column}_{num_of_column}_{rate}.csv"

        df.to_csv(os.path.join(self.filtered_csv_directory, csv_name), index=False)


dataset_name = "S1-ADL1.dat"
num_sensor_labels = 2
poisoning_conf = {
    "attack_labels": ["locomotion"], # Add gesture, location?
    "label_mode": "swim", # swim or swap
    "position": {
        "in_column": "random", # random or consective??
        "num_of_column": 1, # don't over num_sensor_labels.
        "rate": 0.1,
    }
}

def main():
    UCI_data_handler = UCIDataHandler(dataset_name, num_sensor_labels, poisoning_conf)
    df = UCI_data_handler.load_df()
    filtered_df = UCI_data_handler.filter(df)
    poisoned_df = UCI_data_handler.poisoning(filtered_df)
    UCI_data_handler.export_csv(poisoned_df)
    print("Finished.")

if __name__ == "__main__":
    main()

