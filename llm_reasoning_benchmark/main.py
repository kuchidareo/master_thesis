import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import load_datasets
from misclassification import apply_misclassification
from llm_experiment import activity_comprehension_ex, duration_comprehension_ex
from utils import generate_filename, export_csv, export_llm_conversation

@hydra.main(config_path="conf", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    filtered_df = load_datasets(cfg.dataset)
    misclassified_df, misclassified_row_indices = apply_misclassification(filtered_df, cfg.misclassification, cfg.dataset)

    filename = generate_filename(cfg.dataset, cfg.misclassification, cfg.gpt_model, cfg.trial)
    export_csv(misclassified_df, filename)

    csv_text = misclassified_df.to_csv(index=False, header=False)

    if cfg.experiment_type == "household":
        conversation = activity_comprehension_ex(csv_text, cfg.gpt_model, misclassified_df, misclassified_row_indices)
    elif cfg.experiment_type == "measure_activity_duration":
        conversation = duration_comprehension_ex(csv_text, cfg.gpt_model)

    export_llm_conversation(conversation, filename)

if __name__ == "__main__":
    main()

