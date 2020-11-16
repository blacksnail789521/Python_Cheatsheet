score_list = [8, 20, 4, 20]
max_score = max(score_list)

print("---------------------")
print("Find the first match.")
first_match_index = score_list.index(max_score)
print(first_match_index)

print("---------------------")
print("Find all matches.")
match_index_list = [ index for index, element in enumerate(score_list) if element == max_score ]
print(match_index_list)