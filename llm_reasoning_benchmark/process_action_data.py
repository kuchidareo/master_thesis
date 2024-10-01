import os
import random

import hydra
from hydra.utils import get_original_cwd
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from static import UCI_dataset_config

class UCIDataHandler():
    config = UCI_dataset_config

    def __init__(self, original_directory, dataset_name, num_sensor_labels, poisoning_conf) -> None:
        self.dataset_directory = os.path.join(original_directory, self.config["dataset_directory"])
        self.dataset_name = dataset_name
        self.num_sensor_labels = num_sensor_labels
        self.poisoning_conf = poisoning_conf
        self.indices_to_poison = None

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
                self.indices_to_poison = random.sample(range(len(poisoned_df)), num_rows_to_poison)

                for i in self.indices_to_poison:
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


def export_csv(df, filename):
    df.to_csv(f"{filename}.csv", index=False)


def export_llm_conversation(conversation, filename):
    with open(f"{filename}.md", "w") as f:
        for utterance in conversation:
            f.write(f"### {utterance['role']}\n")
            f.write(f"{utterance['content']}\n")


def generate_filename(dataset_name, trial, num_sensor_labels, poisoning_conf):
        attack_labels = str(poisoning_conf["attack_labels"])
        label_mode = poisoning_conf["label_mode"]
        in_column = poisoning_conf["position"]["in_column"]
        num_of_column = poisoning_conf["position"]["num_of_column"]
        rate = poisoning_conf["position"]["rate"]

        filename = f"{dataset_name}_{num_sensor_labels}_{attack_labels}_{label_mode}_{in_column}_{num_of_column}_{rate}_{trial}"
        return filename


def get_answer_from_llm(conversation):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    return response.choices[0].message.content


def llm_experiment(UCI_data_handler, poisoned_df, csv_text):
    conversation = [{"role": "system", "content": "You are a helpful assstant. You'll read sensor label table and analyze what might have happened."}]
    
    first_question = """
        This CSV file provides timestamps along with human activity labels captured by different independent sensors.
        Could you help explain step by step what might have happened and determine what situation the person might be in?
    """

    poisoned_index = random.choice(UCI_data_handler.indices_to_poison)
    relative_time = poisoned_df.loc[poisoned_index, "relative_time(s)"]
    second_question = f"""
        Based on the context, what migh have happend at {relative_time} second?
    """
    
    conversation.append({"role": "user", "content": f"{first_question}\n{csv_text}"})
    first_answer = get_answer_from_llm(conversation)
    conversation.append({"role": "assistant", "content": first_answer})
    conversation.append({"role": "user", "content": second_question})
    second_answer = get_answer_from_llm(conversation)
    conversation.append({"role": "assistant", "content": second_answer})

    return conversation


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@hydra.main(config_path="conf", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    UCI_data_handler = UCIDataHandler(get_original_cwd(), cfg.dataset_name, cfg.num_sensor_labels, cfg.poisoning_conf)
    df = UCI_data_handler.load_df()

    filtered_df = UCI_data_handler.filter(df)
    poisoned_df = UCI_data_handler.poisoning(filtered_df)

    filename = generate_filename(cfg.dataset_name, cfg.trial, cfg.num_sensor_labels, cfg.poisoning_conf)
    export_csv(poisoned_df, filename)

    csv_text = poisoned_df.to_csv(index=False, header=False)
    conversation = llm_experiment(UCI_data_handler, poisoned_df, csv_text, filename)
    export_llm_conversation(conversation, filename)
   

if __name__ == "__main__":
    main()

