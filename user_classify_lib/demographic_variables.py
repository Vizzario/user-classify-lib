
hours_screen_per_day_responses= [
    "0-2",
    "2-5",
    "5-9",
    "9-12",
    "12+"
]

hours_screen_per_day_values= [
    1,
    3.5,
    7,
    10.5,
    14
]

hours_activity_per_week_responses=[
    "0-1",
    "1-2",
    "2-3",
    "3-4",
    "4+"
]

hours_activity_per_week_values=[
    0,
    1,
    2,
    3,
    4
]

vision_prescription=[
    "farsighted",
    "nearsighted",
    "only glasses to read"
]

age_range=[
    "0-15",
    "16-25",
    "26-35",
    "36-45",
    "46-55",
    "56-65",
    "65+"
]

def bin_age_range(age_val: int, age_dict: dict):
    if age_val <= 15:
        age_dict["0-15"] += 1
    elif age_val <= 25:
        age_dict["16-25"] += 1
    elif age_val <= 35:
        age_dict["26-35"] += 1
    elif age_val <= 45:
        age_dict["36-45"] += 1
    elif age_val <= 55:
        age_dict["46-55"] += 1
    elif age_val <= 65:
        age_dict["56-65"] += 1
    else:
        age_dict["65+"] += 1
    return age_dict