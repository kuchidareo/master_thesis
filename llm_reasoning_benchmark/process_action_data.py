import os
import random

import hydra
from hydra.utils import get_original_cwd
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from static import UCI_dataset_config, UCI_locomotion_misclassify_trend, UCI_gesture_misclassify_trend, UCI_household_llm_ex_questions, UCI_activity_duration_measure_llm_ex_questions
from static import UCI_locomotion_label_to_index, UCI_gesture_label_to_index


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

        if self.poisoning_conf["position"]["in_column"] == "random":
            poisoned_df = self.random_poisoning(poisoned_df, attack_labels)
        elif self.poisoning_conf["position"]["in_column"] == "sequential":
            poisoned_df = self.sequential_poisoning(poisoned_df, attack_labels)
        elif self.poisoning_conf["position"]["in_column"] == "real":
            poisoned_df = self.real_poisoning(poisoned_df, attack_labels)

        column_order = sorted(poisoned_df.columns)
        poisoned_df = poisoned_df[column_order]

        return poisoned_df

    def random_poisoning(self, df, attack_labels):
        num_rows_to_poison = int(len(df) * self.poisoning_conf["position"]["rate"])
        self.indices_to_poison = random.sample(range(len(df)), num_rows_to_poison)

        for i in self.indices_to_poison:
            poisoning_column_indices = random.sample(range(self.num_sensor_labels), self.poisoning_conf["position"]["num_of_column"])
            for j in poisoning_column_indices:
                for label in attack_labels:
                    if self.poisoning_conf["label_mode"] == "swim":
                        df.loc[i, f"{label}_label_{j}"] = 'Swim'
                    elif self.poisoning_conf["label_mode"] == "flip":
                        df.loc[i, f"{label}_label_{j}"] = self.generate_flip_label(df, i, label, j)
        return df

    def sequential_poisoning(self, df, attack_labels):
        total_length = len(df)
        num_rows_to_poison = int(total_length * self.poisoning_conf["position"]["rate"])
        sequence_length = self.poisoning_conf["position"]["sequence_length"]
        num_poison_blocks = (num_rows_to_poison // sequence_length) + 1
        num_of_block = total_length // num_poison_blocks
        self.indices_to_poison = []
        poison_counter = 0

        if num_poison_blocks * sequence_length > total_length:
            raise ValueError("Too many elements to poison")

        for block_index in range(num_poison_blocks):
            start_index = random.randint(block_index*num_of_block, (block_index+1)*num_of_block - sequence_length + 1)
            for i in range(start_index, start_index + sequence_length):
                if poison_counter >= num_rows_to_poison:
                    break
                for j in range(self.poisoning_conf["position"]["num_of_column"]):
                    for label in attack_labels:
                        if self.poisoning_conf["label_mode"] == "swim":
                            df.loc[i, f"{label}_label_{j}"] = 'Swim'
                        elif self.poisoning_conf["label_mode"] == "flip":
                            df.loc[i, f"{label}_label_{j}"] = self.generate_flip_label(df, i, label, j)
                self.indices_to_poison.append(i)
                poison_counter += 1
        return df

    def real_poisoning(self, df, attack_labels):
        non_zero_locomotion_rows = len(df[df['locomotion_label_0'] != "0"])
        print(f"Number of rows where 'locomotion_label' is not 0: {non_zero_locomotion_rows}")
        non_zero_gesture_rows = len(df[df['gesture_label_0'] != "0"])
        print(f"Number of rows where 'gesture_label' is not 0: {non_zero_gesture_rows}")

        num_rows_to_locomotion_poison = int(non_zero_locomotion_rows * self.poisoning_conf["position"]["rate"])
        num_rows_to_gesture_poison = int(non_zero_gesture_rows * self.poisoning_conf["position"]["rate"])
        
        columns_to_poison = random.sample(
            range(self.num_sensor_labels), 
            self.poisoning_conf["position"]["num_of_column"]
        )

        for column_index in columns_to_poison:
            for label in attack_labels:
                if label == "locomotion":
                    misclassify_trend = UCI_locomotion_misclassify_trend
                    attack_labels = list(UCI_locomotion_label_to_index.keys())
                    num_rows_to_poison = num_rows_to_locomotion_poison
                elif label == "gesture":
                    misclassify_trend = UCI_gesture_misclassify_trend
                    attack_labels = list(UCI_gesture_label_to_index.keys())
                    num_rows_to_poison = num_rows_to_gesture_poison

                misclassify_trend_rate = [row[2] for row in misclassify_trend]
                misclassifications = random.choices(
                    misclassify_trend,
                    weights=misclassify_trend_rate, 
                    k=num_rows_to_poison
                )
                if self.poisoning_conf["label_mode"] == "real":
                    for attack_label, change_label, _ in misclassifications:
                        self.apply_label_change(df, label, column_index, attack_label, change_label, attack_labels)
                elif self.poisoning_conf["label_mode"] == "fuse":
                    for attack_label, _, _ in misclassifications:
                        if label == "locomotion":
                            change_label = "Swim"
                        elif label == "gesture":
                            change_label = "Shake Hands"
                        self.apply_label_change(df, label, column_index, attack_label, change_label, attack_labels)
                        
        return df

    def duplicate_attack_columns(self, df, attack_labels):
        for label in attack_labels:
            for i in range(1, self.num_sensor_labels):
                df[f"{label}_label_{i}"] = df[f"{label}_label_0"].copy()
        return df

    def generate_flip_label(self, df, index, label, column):
        # Random Flipping
        flipped_label = random.choice(list(self.config["locomotion_label_legend"].values()))
        while flipped_label == df.loc[index, f"{label}_label_{column}"]:
            flipped_label = random.choice(list(self.config["locomotion_label_legend"].values()))

        return flipped_label

    def apply_label_change(self, df, label, column_index, attack_label, change_label, attack_labels):
        is_attacked = False
        while not is_attacked:
            target_indexes = df[df[f"{label}_label_{column_index}"] == attack_label].index
            if not target_indexes.empty:
                target_index = random.choice(target_indexes)
                df.loc[target_index, f"{label}_label_{column_index}"] = change_label
                print(f"index: {target_index}, {label}_label_{column_index}: {attack_label} -> {change_label}")
                is_attacked = True
            else:
                print(f"No instances of {attack_label} found in column {label}_label_{column_index}")
                attack_label = random.choice(attack_labels)
                while attack_label == "0":
                    attack_label = random.choice(attack_labels)


def export_csv(df, filename):
    df.to_csv(f"{filename}.csv", index=False)


def export_llm_conversation(conversation, filename):
    with open(f"{filename}.md", "w") as f:
        for utterance in conversation:
            f.write(f"### {utterance['role']}\n")
            f.write(f"{utterance['content']}\n")


def generate_filename(dataset_name, gpt_model, trial, num_sensor_labels, poisoning_conf):
        attack_labels = str(poisoning_conf["attack_labels"])
        label_mode = poisoning_conf["label_mode"]
        in_column = poisoning_conf["position"]["in_column"]
        num_of_column = poisoning_conf["position"]["num_of_column"]
        rate = poisoning_conf["position"]["rate"]

        filename = f"{dataset_name}_{gpt_model}_{num_sensor_labels}_{attack_labels}_{label_mode}_{in_column}_{num_of_column}_{rate}_{trial}"
        return filename


def get_answer_from_llm(conversation, gpt_model):
    response = client.chat.completions.create(
        model=gpt_model,
        messages=conversation
    )

    return response.choices[0].message.content

def llm_experiment(gpt_model, questions):
    conversation = [{"role": "system", "content": "You are a helpful assstant. You'll read sensor label table and analyze what might have happened."}]
    
    for question in questions:
        conversation.append({"role": "user", "content": question})
        answer = get_answer_from_llm(conversation, gpt_model)
        conversation.append({"role": "assistant", "content": answer})

    return conversation

def household_experiment(csv_text, gpt_model, poisoned_df, UCI_data_handler):
    first_question = f"{UCI_household_llm_ex_questions[0]}\n{csv_text}"
    poisoned_index = random.choice(UCI_data_handler.indices_to_poison) if UCI_data_handler.indices_to_poison else 0
    relative_time = poisoned_df.loc[poisoned_index, "relative_time(s)"]
    second_question = f"{UCI_household_llm_ex_questions[1]} at {relative_time} seconds."

    questions = [first_question, second_question]
    conversation = llm_experiment(gpt_model, questions)

    return conversation

def measure_activity_duration(csv_text, gpt_model):
    first_question = f"{UCI_activity_duration_measure_llm_ex_questions[0]}\n{csv_text}"
    questions = [first_question]
    conversation = llm_experiment(gpt_model, questions)

    return conversation


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@hydra.main(config_path="conf", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    UCI_data_handler = UCIDataHandler(get_original_cwd(), cfg.dataset_name, cfg.num_sensor_labels, cfg.poisoning_conf)
    df = UCI_data_handler.load_df()
    # activity_col_idx = UCI_dataset_config["HL_Activity_column_index"]
    # activity_df = df.iloc[:, activity_col_idx]
    # print(activity_df.value_counts())

    filtered_df = UCI_data_handler.filter(df)
    poisoned_df = UCI_data_handler.poisoning(filtered_df)

    filename = generate_filename(cfg.dataset_name, cfg.gpt_model, cfg.trial, cfg.num_sensor_labels, cfg.poisoning_conf)
    export_csv(poisoned_df, filename)

    csv_text = poisoned_df.to_csv(index=False, header=False)

    if cfg.experiment_type == "household":
        conversation = household_experiment(csv_text, cfg.gpt_model, poisoned_df, UCI_data_handler)
    elif cfg.experiment_type == "measure_activity_duration":
        conversation = measure_activity_duration(csv_text, cfg.gpt_model)

    export_llm_conversation(conversation, filename)

if __name__ == "__main__":
    main()

