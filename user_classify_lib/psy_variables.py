
basic_variables= [
    'blur_errors',
    'brightness_errors',
    'contrast_errors',
    'dimensions_errors',
    'h_position_error',
    'hue_errors',
    'proportion_correct',
    'response_rate',
    'rt_mode',
    'rt_width',
    'saturation_errors',
    'time_on_task',
    'v_position_error',
]

eccentricity = [
    'pc_eccentricity_slope',
    'pc_eccentricity_threshold',
    'rt_eccentricity_slope',
]

central = [
    'h_pe_central',
    'pc_central',
    'rt_central_mode',
    'rt_central_width',
    'v_pe_central'
]

peripheral = [
    'h_pe_peripheral',
    'pc_peripheral',
    'rt_peripheral_mode',
    'rt_peripheral_width',
    'v_pe_peripheral',
]

speed = [
    'pc_speed_slope',
    'pc_speed_threshold',
    'rt_speed_slope',
]

duration = [
    'pc_duration_slope',
    'pc_duration_threshold',
    'rt_duration_slope'
]
concurrency = [
    'pc_concurrent_stimuli_slope',
    'pc_concurrent_stimuli_threshold',
    'rt_concurrent_stimuli_slope'
]

focussed = [
    'h_pe_focussed',
    'pc_focussed',
    'rt_focussed_mode',
    'rt_focussed_width',
    'v_pe_focussed'
]

divided = [
    'h_pe_divided',
    'pc_divided',
    'rt_divided_mode',
    'rt_divided_width',
    'v_pe_divided'
]

color = [
    'pc_color_slope',
    'pc_color_threshold',
    'rt_color_slope'
]

contrast = [
    'pc_contrast_slope',
    'pc_contrast_threshold',
    'rt_contrast_slope'
]

size = [
    'pc_size_slope',
    'pc_size_threshold',
    'rt_size_slope'
]
sub_components= [
    'accuracy_reaction',
    'accuracy_targeting',
    'detection_acuity',
    'detection_color',
    'detection_contrast',
    'endurance_fatigue',
    'field_of_view_central',
    'field_of_view_peripheral',
    'multi_tracking_divided',
    'multi_tracking_focussed'
]

components = [
    'accuracy_combined',
    'detection_combined',
    'endurance_combined',
    'field_of_view_combined',
    'multi_tracking_combined'
]

overall = [
    'overall'
]

psy_range=[
    "055-065",
    "066-075",
    "076-085",
    "086-095",
    "096-105",
    "106-115",
    "116-125",
    "126-135",
    "136-145"
]

def bin_psy_range(psy_val: float, psy_dict: dict):
    if psy_val <=65:
        psy_dict["055-065"] += 1
    elif psy_val <= 75:
        psy_dict["066-075"] += 1
    elif psy_val <= 85:
        psy_dict["076-085"] += 1
    elif psy_val <= 95:
        psy_dict["086-095"] += 1
    elif psy_val <= 105:
        psy_dict["096-105"] += 1
    elif psy_val <= 115:
        psy_dict["106-115"] += 1
    elif psy_val <= 125:
        psy_dict["116-125"] += 1
    elif psy_val <= 135:
        psy_dict["126-135"] += 1
    else:
        psy_dict["136-145"] += 1
    return psy_dict