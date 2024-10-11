UCI_dataset_config = {
    "dataset_directory": "OpportunityUCIDataset/dataset",
    "sampling_frequency_hz": 32,
    "locomotion_column_index": 243, # Read dataset/column_names.txt, label_legend.txt
    "HL_Activity_column_index": 244,
    "LL_Left_Arm_column_index": 245,
    "LL_Left_Arm_Object_column_index": 246,
    "LL_Right_Arm_column_index": 247,
    "LL_Right_Arm_Object_column_index": 248,
    "ML_Both_Arms_column_index": 249,
    "locomotion_label_legend": {
        "1": "Stand",
        "2": "Walk",
        "4": "Sit",
        "5": "Lie"
    },
    "activity_label_legend": {
        "101": "Relaxing",
        "102": "Coffee time",
        "103": "Early morning",
        "104": "Cleanup",
        "105": "Sandwich time",
    },
    "ML_Both_Arms_label_legend": {
        "406516": "Open Door 1",
        "406517": "Open Door 2",
        "404516": "Close Door 1",
        "404517": "Close Door 2",
        "406520": "Open Fridge",
        "404520": "Close Fridge",
        "406505": "Open Dishwasher",
        "404505": "Close Dishwasher",
        "406519": "Open Drawer 1",
        "404519": "Close Drawer 1",
        "406511": "Open Drawer 2",
        "404511": "Close Drawer 2",
        "406508": "Open Drawer 3",
        "404508": "Close Drawer 3",
        "408512": "Clean Table",
        "407521": "Drink from Cup",
        "405506": "Toggle Switch"
    },
}
UCI_ex_result_annotation = {
    "story_correction": {
        "gpt-4o-2024-08-06": {
            0.0: [True, True],
            0.1: [True, True, False, True, True, True, True, True, True, True],
            0.2: [True, True, True, False, True, True, True, True, True, True],
            0.3: [True, True, True, True, True, True, True, True, False, True],
            0.4: [False, False, False, True, True, False, False, False, True, False],
            0.5: [True, True, False, True, False, True, False, True, True, True],
            0.6: [False, False, False, True, False, False, True, False, False, False],
            0.7: [False, True, False, False, False, False, True, False, False, True]
        },
        "gpt-4o-mini": {
            0.0: [True, True],
            0.1: [False, False, False, False, False, False, False, False, False, False],
            0.2: [False, False, False, False, False, False, False, False, False, False],
            0.3: [False, False, False, False, False, False, False, False, False, False],
            0.4: [False, False, False, False, False, False, False, False, False, False],
            0.5: [False, False, False, False, False, False, False, False, False, False],
            0.6: [False, False, False, False, False, False, False, False, False, False],
            0.7: [False, False, False, False, False, False, False, False, False, False]
        },
        "gpt-3.5-turbo": {
            0.0: [True, True],
            0.1: [False, False, False, False, False, False, False, False, False, False],
            0.2: [False, False, False, False, False, False, False, False, False, False],
            0.3: [False, False, False, False, False, False, False, False, False, False],
            0.4: [False, False, False, False, False, False, False, False, False, False],
            0.5: [False, False, False, False, False, False, False, False, False, False],
            0.6: [False, False, False, False, False, False, False, False, False, False],
            0.7: [False, False, False, False, False, False, False, False, False, False]
        }
    },
    "binary_choice": {
        "gpt-4o-2024-08-06": {
            0.0: [True, True],
            0.1: [True, True, False],
            0.3: [True, True, True],
            0.5: [True, True, False],
            0.7: [False, True, False],
            1.0: [True, True, True]
        }
    }
}
UCI_index_to_locomotion_label = {
    0: "None",
    1: "Stand",
    2: "Walk",
    3: "Sit",
    4: "Lie"
}
UCI_locomotion_label_to_index = {
    "None": 0,
    "Stand": 1,
    "Walk": 2,
    "Sit": 3,
    "Lie": 4
}
UCI_index_to_gesture_label = {
    0: "None",
    1: "Open Door 1",
    2: "Open Door 2",
    3: "Close Door 1",
    4: "Close Door 2",
    5: "Open Fridge",
    6: "Close Fridge",
    7: "Open Dishwasher",
    8: "Close Dishwasher",
    9: "Open Drawer 1",
    10: "Close Drawer 1",
    11: "Open Drawer 2",
    12: "Close Drawer 2",
    13: "Open Drawer 3",
    14: "Close Drawer 3",
    15: "Clean Table",
    16: "Drink from Cup",
    17: "Toggle Switch"
}
UCI_gesture_label_to_index = {
    "None": 0,
    "Open Door 1": 1,
    "Open Door 2": 2,
    "Close Door 1": 3,
    "Close Door 2": 4,
    "Open Fridge": 5,
    "Close Fridge": 6,
    "Open Dishwasher": 7,
    "Close Dishwasher": 8,
    "Open Drawer 1": 9,
    "Close Drawer 1": 10,
    "Open Drawer 2": 11,
    "Close Drawer 2": 12,
    "Open Drawer 3": 13,
    "Close Drawer 3": 14,
    "Clean Table": 15,
    "Drink from Cup": 16,
    "Toggle Switch": 17
}

def calculate_misclassification_trend(confusion_matrix, index_to_label):
    regular_matrix = [[0 for _ in range(len(confusion_matrix))] for _ in range(len(confusion_matrix[0]))]
    misclassify_trend = []
    total_proportion = 0

    for i, row in enumerate(confusion_matrix):
        num_total = sum(row)
        for j, value in enumerate(row):
            if i == j:
                regular_matrix[i][j] = 0
            else:
                regular_matrix[i][j] = value / num_total
                total_proportion += value / num_total

    for i, row in enumerate(regular_matrix):
        for j, value in enumerate(row):
            probability = value / total_proportion
            if probability > 0.0:
                misclassify_trend.append([index_to_label[i], index_to_label[j], probability])
    return misclassify_trend

UCI_cnn_locomotion_confusion_matrix = [
    [1200, 0, 0, 0, 0],
    [2, 1188, 23, 0, 0],
    [1, 0, 706, 11, 0],
    [0, 0, 3, 589, 1],
    [0, 0, 0, 1, 247],
]
UCI_cnn_gesture_confusion_matrix = [
    [4297, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 96, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 79, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 35, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 30, 1, 0, 0, 0, 5, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 17, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 24, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 205, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 20],
]
UCI_locomotion_misclassify_trend = calculate_misclassification_trend(UCI_cnn_locomotion_confusion_matrix, UCI_index_to_locomotion_label)
UCI_gesture_misclassify_trend = calculate_misclassification_trend(UCI_cnn_gesture_confusion_matrix, UCI_index_to_gesture_label)
# print(UCI_locomotion_misclassify_trend)
print(UCI_gesture_misclassify_trend)
UCI_activity_durations = {
    "Relaxing": 57,
    "Coffee-time": 178,
    "Early-morning": 349,
    "Cleanup": 160,
    "Sandwich-time": 319,
}

UCI_household_llm_ex_questions = [
    """
This CSV file provides timestamps along with human activity labels captured by different independent sensors.
Could you help explain step by step what might have happened and determine what situation the person might be in?
    """,
    """
Based on the context, what might have happened at
    """,
]
UCI_activity_duration_measure_llm_ex_questions = [
    """
This CSV file provides timestamps along with human activity labels captured by different independent sensors.
Calculate the duration of each activity: Relaxing, Coffee time, Early morning, Cleanup, and Sandwich time.
    """,
]