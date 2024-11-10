# System Imports
from typing import Callable

# Third party imports
import pandas as pd

def get_neighbors(symbol_data, row, params, n):
    # Initialize the condition to True so we can apply multiple conditions using & operator
    condition = True
    for param, step in params.items():
        condition &= symbol_data[param].between(row[param] - (n * step), row[param] + (n * step))
    # Filter based on the dynamic condition
    neighbors = symbol_data[condition]
    return neighbors

def find_best_params(metrics_df: pd.DataFrame, 
                     target: str, params: dict[str, int], 
                     n:int=1, agg_func:Callable[[pd.Series], float]=pd.Series.median
                     ) -> dict[str, tuple]:
  '''
  Finds the best n-neighborhood for a specific target parameters,
    where n=1 is the best single point
  '''
  # Create a dictionary to store the best triple for each symbol
  best_neighborhood = {}

  # Iterate through each symbol in the DataFrame
  for symbol in metrics_df['symbol'].unique():
    # Filter the DataFrame to only include rows for the current symbol
    symbol_data = metrics_df[metrics_df['symbol'] == symbol]

    best_avg_target = float('-inf')
    curr_best = None

    # Iterate through each row for the current symbol
    for _, row in symbol_data.iterrows():
      # Get the neighboring points for the current row, considering n neighbors
      # TODO: Make this work for other strategies (get strat params as params to this function)
      # neighbors = symbol_data
      # for param, step in params.items():
      #   neighbors = neighbors[(symbol_data[param].between(row[param] - (step*n), row[param] + (step*n)))]

      neighbors = get_neighbors(symbol_data, row, params, n)

      # Calculate the average CAGR of the current point and its neighbors
      avg_target = agg_func(neighbors[target])

      # Update the best triple if the current average CAGR is higher
      if avg_target > best_avg_target:
        best_avg_target = avg_target
        curr_best = {param: row[param] for param in params.keys()}

    # Store the best triple for the current symbol
    best_neighborhood[symbol] = curr_best

  return best_neighborhood