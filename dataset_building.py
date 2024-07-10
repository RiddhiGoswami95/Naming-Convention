import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/final_sections_names.csv', encoding='latin-1')
names1 = df['Sections'].tolist()
names1 = [str(x) for x in names1]

names_final1 = []
names_final2 = []
count = 0
for name in names1:
  print(str(count) + "/ 21681")
  count += 1
  for i in range(20):
    pos = np.random.randint(0, len(name) - 1)
    if name[pos].isdigit() or name[pos].isalpha():
      if name[pos].isupper():
        name_modified = name[:pos] + name[pos].lower() + name[pos + 1:]
      else:
        name_modified = name[:pos] + name[pos].upper() + name[pos + 1:]
    else:
      name_modified = name[:pos] + '_' + name[pos + 1:]

    # Additional Conditions for Misspellings
    if np.random.rand() < 0.2:  # 20% chance of adding extra character
      if np.random.rand() < 0.5:
        name_modified = name_modified[:-1] + chr(np.random.randint(
            97, 123)) + name_modified[-1]  # Add lowercase letter
      else:
        name_modified = name_modified[:-1] + chr(np.random.randint(
            65, 91)) + name_modified[-1]  # Add uppercase letter
    if np.random.rand() < 0.1:  # 10% chance of deleting a character
      if len(name_modified) > 1:
        pos = np.random.randint(0, len(name_modified) - 1)
        name_modified = name_modified[:pos] + name_modified[pos + 1:]
    if np.random.rand(
    ) < 0.1:  # 10% chance of replacing a character with a random one
      pos = np.random.randint(0, len(name_modified) - 1)
      if name_modified[pos].isalpha():
        name_modified = name_modified[:pos] + chr(np.random.randint(
            97, 123)) + name_modified[pos + 1:]
      else:
        name_modified = name_modified[:pos] + str(np.random.randint(
            0, 10)) + name_modified[pos + 1:]

    if name_modified not in names_final2:
      names_final1.append(name)
      names_final2.append(name_modified)
    else:
      pos = np.random.randint(0, len(name_modified) - 1)
      if name_modified[pos].isdigit() or name_modified[pos].isalpha():
        if name_modified[pos].isupper():
          name_modified = name_modified[:pos] + name_modified[pos].lower(
          ) + name_modified[pos + 1:]
        else:
          name_modified = name_modified[:pos] + name_modified[pos].upper(
          ) + name_modified[pos + 1:]
      else:
        name_modified = name_modified[:pos] + '_' + name_modified[pos + 1:]
      names_final1.append(name)
      names_final2.append(name_modified)

print(len(names_final1), len(names_final2))
df2 = pd.DataFrame({'Incorrect': names_final2, 'Correct': names_final1})

df2.to_csv('Datasets/dataset_final.csv', index=False)
