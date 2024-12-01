# My_vars.py
class vars:
   warn_files = {
    '雷注意報': True,
    '大雨注意報': True,
    '洪水注意報': True,
    '高潮注意報': False,
    '強風注意報': True,
    '波浪注意報': False,
    '大雨警報': True,
    '洪水警報': True,
    '暴風警報': True,
    '高潮警報': False,
    '波浪警報': False,
    '大雨特別警報': True
    }
#    warn_files = {
#     '雷注意報': False,
#     '大雨注意報': False,
#     '洪水注意報': False,
#     '高潮注意報': False,
#     '強風注意報': False,
#     '波浪注意報': False,
#     '大雨警報': False,
#     '洪水警報': False,
#     '暴風警報': True,
#     '高潮警報': False,
#     '波浪警報': False,
#     '大雨特別警報': True
#     }
   

    # warn_files = {
    #     '雷注意報': 'Thunderstorm Advisory',
    #     '大雨注意報': 'Heavy Rain Advisory',
    #     '洪水注意報': 'Flood Advisory',
    #     '高潮注意報': 'Storm Surge Advisory',
    #     '強風注意報': 'Strong Wind Advisory',
    #     '波浪注意報': 'High Wave Advisory',
    #     '大雨警報': 'Heavy Rain Warning',
    #     '洪水警報': 'Flood Warning',
    #     '暴風警報': 'Storm Warning',
    #     '高潮警報': 'Storm Surge Warning',
    #     '波浪警報': 'High Wave Warning',
    #     '大雨特別警報': 'Heavy Rain Emergency Warning'
    # }


def get_warning_list():
    return vars.warn_files