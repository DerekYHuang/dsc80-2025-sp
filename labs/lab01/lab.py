# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.21')


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    nums_sorted = sorted(nums)
    n = len(nums_sorted)
    mean = sum(nums) / n
    
    if n % 2 == 1:
        median = nums_sorted[n // 2]
    else:
        median = (nums_sorted[n // 2 - 1] + nums_sorted[n // 2]) / 2
    
    return median <= mean


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    prefixes = [s[:i] for i in range(1, n+1)]
    reversed_prefixes = prefixes[::-1]
    result = ''.join(reversed_prefixes)
    return result


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    if not ints:
        return []
    
    max_num = max(num + n for num in ints)
    min_num = min(num - n for num in ints)
    max_digits = len(str(max(max_num, min_num)))
    result = []
    
    for num in ints:
        start = num - n
        end = num + n
        exploded = range(start, end + 1)
        exploded_strs = [str(x).zfill(max_digits) for x in exploded]
        result.append(' '.join(exploded_strs))
    
    return result


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
    result = []
    for line in fh:
        stripped_line = line.rstrip('\n')
        if stripped_line:
            last_char = stripped_line[-1]
            result.append(last_char)
    return ''.join(result)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    indices = np.arange(A.size)
    sqrt_indices = np.sqrt(indices)
    return A + sqrt_indices

def where_square(A):
    sqrt_A = np.sqrt(A)
    return sqrt_A == np.floor(sqrt_A)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    selected_columns = []
    
    for col in range(num_cols):
        column_sum = 0.0
        for row in range(num_rows):
            column_sum += matrix[row][col]
        column_mean = column_sum / num_rows
        if column_mean > cutoff:
            selected_columns.append(col)
    
    result = []
    for row in range(num_rows):
        new_row = [matrix[row][col] for col in selected_columns]
        result.append(new_row)
    
    return np.array(result)


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    column_means = np.mean(matrix, axis=0)
    mask = column_means > cutoff
    return matrix[:, mask]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    return np.round((A[1:] - A[:-1]) / A[:-1], 2)

def with_leftover(A):
    daily_leftover = 20 % A  
    total_leftover = np.cumsum(daily_leftover)  
    affordable = total_leftover >= A  
    if np.any(affordable):
        return int(np.argmax(affordable))
    return -1


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    num_players = len(salary)
    num_teams = salary['Team'].nunique()
    total_salary = salary['Salary'].sum()
    
    idx_max = salary['Salary'].idxmax()
    highest_salary = salary.loc[idx_max, 'Player']
    
    avg_los = round(salary[salary['Team'] == 'Los Angeles Lakers']['Salary'].mean(), 2)
    
    fifth_row = salary.sort_values('Salary').iloc[4]
    fifth_lowest = f"{fifth_row['Player']}, {fifth_row['Team']}"
    
    def clean_last(name):
        return name.replace(' Jr.', '').replace(' III', '').split()[-1]
    
    last_names = salary['Player'].apply(clean_last)
    duplicates = last_names.duplicated().any()
    
    team_highest = salary.loc[idx_max, 'Team']
    total_highest = salary[salary['Team'] == team_highest]['Salary'].sum()
    
    return pd.Series({
        'num_players': num_players,
        'num_teams': num_teams,
        'total_salary': total_salary,
        'highest_salary': highest_salary,
        'avg_los': avg_los,
        'fifth_lowest': fifth_lowest,
        'duplicates': duplicates,
        'total_highest': total_highest
    })



# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    rows = []

    with open(fp, 'r') as f:
        next(f)  
        for line in f:
            line = line.strip()
            if not line:
                continue

            last_quote = line.rfind('"')
            first_quote = line.rfind('"', 0, last_quote)
            geo = line[first_quote + 1: last_quote] if first_quote != -1 and last_quote != -1 else "0.0,0.0"

            
            lat_lon = geo.split(',')
            if len(lat_lon) != 2:
                lat, lon = 0.0, 0.0
            else:
                try:
                    lat = float(lat_lon[0])
                    lon = float(lat_lon[1])
                except:
                    lat, lon = 0.0, 0.0

            before_geo = line[:first_quote].strip(', ') if first_quote != -1 else line

            parts = [x.strip().strip('"') for x in before_geo.split(',')]

            
            first = parts[0] if len(parts) > 0 else "Unknown"
            last = parts[1] if len(parts) > 1 else "Unknown"

            try:
                weight = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
            except:
                weight = 0.0

            try:
                height = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
            except:
                height = 0.0

            rows.append({
                'first': first,
                'last': last,
                'weight': weight,
                'height': height,
                'geo': f"{lat},{lon}"
            })

    return pd.DataFrame(rows)
