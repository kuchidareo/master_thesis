import csv
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

from static import UCI_ex_result_annotation

# Import the all csv data from story_correction/cosine_similarity dir.
# The file name contains the poisoning rate. e.g. story_correction_text-embedding-3-small_gpt-4o-2024-08-06_0.1.csv
def import_csv_data(ex_tag):
    data = {}
    for file in os.listdir(f"{ex_tag}/cosine_similarity"):
        model = file.split("_")[3]
        poisoning_rate = file.split("_")[-1].replace(".csv", "")
        if model not in data:
            data[model] = {}
        if poisoning_rate not in data[model]:
            data[model][poisoning_rate] = []

        if file.endswith(".csv"):
            with open(f"{ex_tag}/cosine_similarity/{file}", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    data[model][poisoning_rate].append(row)
        data[model] = {k: v for k, v in sorted(data[model].items(), key=lambda item: item[0])}
        
    return data

# import embedding data model -> poisoning_rate -> baseline or sentence -> embedding
# the embedding list is ordered by the trial number.
def import_embedding_data(ex_tag):
    data = {}
    for file in os.listdir(f"{ex_tag}/embedding"):
        model = file.split("_")[3]
        embedding_model = file.split("_")[2]
        poisoning_rate = file.split("_")[4]
        trial_number = file.split("_")[-2]
        if model not in data:
            data[model] = {}
        if poisoning_rate not in data[model]:
            data[model][poisoning_rate] = {}
        if "sentence" not in data[model][poisoning_rate]:
            data[model][poisoning_rate]["sentence"] = []
        if "baseline" not in data[model][poisoning_rate]:
            data[model][poisoning_rate]["baseline"] = []

        if file.endswith(".pt"):
            if trial_number == "baseline":
                data[model][poisoning_rate]["baseline"] = torch.load(f"{ex_tag}/embedding/{file}")
            else:
                # Append the loaded tensor and sort the list by trial_number
                data[model][poisoning_rate]["sentence"].append((trial_number, embedding_model, torch.load(f"{ex_tag}/embedding/{file}")))
                data[model][poisoning_rate]["sentence"].sort(key=lambda x: x[0])  # Sort by trial_number

    # Make a sorted list from data. Ordered by model -> poisoning_rate -> sentence
    model_order = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o-2024-08-06"]
    returning_list = []
    returning_list.append(data["gpt-3.5-turbo"]["0.1"]["baseline"])
    for model in model_order:
        for rate in data[model].keys():
            for sentence in data[model][rate]["sentence"]:
                returning_list.append(sentence[2])
    
    index_to_content = {}
    index_to_content[0] = "baseline"
    index = 1

    for model in model_order:
        for rate in data[model].keys():
            for sentence in data[model][rate]["sentence"]:
                index_to_content[index] = f"{ex_tag}_{sentence[1]}_{model}_{rate}_{sentence[0]}"
                index += 1
    
    return returning_list, index_to_content

# Make box figure of accuracy on the different rate of poisoning .
def make_cosine_similarity_box_figure():
    ex_tag = "story_correction"
    model = "gpt-4o-2024-08-06"
    data = import_csv_data(ex_tag)[model]
    cosine_similarity = {rate: [] for rate in data.keys()}  # Initialize with empty lists for each rate
    for rate, rows in data.items():
        for row in rows:
            if row[0] == "Trial" or row[0] == "baseline":
                continue
            cosine_similarity[rate].append(float(row[2]))  # Collect cosine similarities for each rate

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(cosine_similarity.values())
    ax.set_xticklabels([f"{rate}" for rate in cosine_similarity.keys()])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity of Story Correction")
    plt.savefig("box_figure.png")
    plt.show()

# Make a line plot of accuracy on the story_correction ex.
def make_line_plot_of_accuracy():
    ex_tag = "story_correction"
    data = import_csv_data(ex_tag)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model, rows in data.items():
        accuracy = {rate: [] for rate in rows.keys()}  # Initialize with empty lists for each rate
        for rate, rows in rows.items():
            for row in rows:
                if row[0] == "Trial" or row[0] == "baseline":
                    continue
                if row[1] == "blue":
                    accuracy[rate].append(1)
                elif row[1] == "red":
                    accuracy[rate].append(0)
            accuracy[rate] = sum(accuracy[rate]) / len(accuracy[rate])  # Calculate accuracy for each rate
        ax.plot(accuracy.keys(), accuracy.values(), "-o", label=model)

    ax.set_xlabel("Poisoning Rate")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("Accuracy of Story Correction")
    plt.savefig("line_acc_plot_all_models.png")
    plt.show()

def make_line_plot_of_cosine_similarity():
    ex_tag = "story_correction"
    data = import_csv_data(ex_tag)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model, rows in data.items():
        cosine_similarity = {rate: [] for rate in rows.keys()}
        for rate, rows in rows.items():
            for row in rows:
                if row[0] == "Trial" or row[0] == "baseline":
                    continue
                cosine_similarity[rate].append(float(row[2]))
            cosine_similarity[rate] = sum(cosine_similarity[rate]) / len(cosine_similarity[rate])
        ax.plot(cosine_similarity.keys(), cosine_similarity.values(), "-o", label=model)
    ax.set_xlabel("Poisoning Rate")
    ax.set_ylabel("Cosine Similarity")
    ax.legend()
    ax.set_title("Cosine Similarity of Story Correction")
    plt.savefig("line_plot_all_models.png")
    plt.show()

def make_pca_plot():
    ex_tag = "story_correction"
    embedding, index_to_content = import_embedding_data(ex_tag)
    import pdb; pdb.set_trace()
    pca = PCA(n_components=2)
    pca.fit(embedding)
    pca_list = pca.transform(embedding)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    baseline = pca_list[0]
    ax.scatter(baseline[0], baseline[1], c="blue")
    ax.annotate(f'baseline', (baseline[0], baseline[1]))

    # index_to_content = {1: 'story_correction_text-embedding-3-small_gpt-3.5-turbo_0.1_1}...
    for i in range(1, len(pca_list)):
        content = index_to_content[i]
        model = content.split("_")[3]
        rate = content.split("_")[4]
        trial = content.split("_")[-1]
        annotation = UCI_ex_result_annotation[ex_tag][model][float(rate)]

        color = "blue" if annotation[int(trial)-1] else "red"
        ax.scatter(pca_list[i][0], pca_list[i][1], c=color)
        ax.annotate(f'{model}', (pca_list[i][0], pca_list[i][1]))

    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("PCA of Story Correction")
    plt.savefig("pca_plot.png")
    plt.show()



def main():
    # make_cosine_similarity_box_figure()
    # make_line_plot_of_accuracy()
    # make_line_plot_of_cosine_similarity()
    make_pca_plot()

if __name__ == "__main__":
    main()