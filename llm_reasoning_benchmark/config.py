UCI_dataset_config = {
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