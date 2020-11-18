import importlib


def get_String_Matcher():
    
    try:
        # https://github.com/ztane/python-Levenshtein
        String_Matcher = getattr(importlib.import_module("Levenshtein.StringMatcher"), "StringMatcher")
        print("String_Matcher: Levenshtein")
    except Exception as e:
        print(e)
        String_Matcher = getattr(importlib.import_module("difflib"), "SequenceMatcher")
        print("String_Matcher: difflib")
    
    return String_Matcher


def get_factor_name_match_info(factor_name_x, factor_name_y, String_Matcher = get_String_Matcher()):
    
    factor_name_x_match_info, factor_name_y_match_info = [], []
    
    blocks = String_Matcher(None, factor_name_x, factor_name_y).get_matching_blocks()
    for index, block in enumerate(blocks):
        
        # The last block is useless.
        if index == len(blocks) - 1:
            break
        
        x_start_index = block[0]
        y_start_index = block[1]
        match_length =  block[2]
        
        factor_name_x_match_info.append({"start_index": x_start_index, "end_index": x_start_index + match_length, \
                                         "match_content": factor_name_x[ x_start_index : x_start_index + match_length ]})
        factor_name_y_match_info.append({"start_index": y_start_index, "end_index": y_start_index + match_length, \
                                         "match_content": factor_name_y[ y_start_index : y_start_index + match_length ]})
    
    return factor_name_x_match_info, factor_name_y_match_info


if __name__ == "__main__":
    
    factor_name_x = "My_name_is_Jason"
    factor_name_y = "Your_name_is_not_Jason"
    factor_name_x_match_info, factor_name_y_match_info = \
        get_factor_name_match_info(factor_name_x, factor_name_y)