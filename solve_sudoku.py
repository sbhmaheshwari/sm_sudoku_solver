'''refer to https://medium.com/activating-robotic-minds/peter-norvigs-sudoku-solver-25779bb349ce'''

def cross(a, b):
  return([i+j for i in a for j in b])
digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
squares = cross(rows,cols)
unitlist = ([cross(rows, c) for c in cols]+
            [cross(r, cols) for r in rows]+
            [cross(i, j) for i in ('ABC', 'DEF', 'GHI') for j in ('123', '456', '789')])
units = {s: [v for v in unitlist if s in v] for s in squares}
peers = {s: set(sum(units[s], []))-set([s]) for s in squares}

def test():
    "A set of unit tests."
    assert len(squares) == 81
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    assert units['C2'] == [
      ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
      ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
      ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == set(
      ['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
       'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
       'A1', 'A3', 'B1', 'B3'])
    print('All tests pass.')
#test()

def grid_values(grid):
  vals = [c for c in grid if c in digits or c in '0.']
  assert len(vals) == 81
  return(dict(zip(squares, vals)))

def parse_grid(grid):
  values = {s: digits for s in squares}
  for s, d in grid_values(grid).items():
    if d in digits and not assign(values, s, d):
      return(False)
  return(values)

def assign(values, s, d):
  othervals = values[s].replace(d, '')
  if all(eliminate(values, s, vals) for vals in othervals):
    return(values)
  else:
    return(False) 

def eliminate(values, s, d):
  if d not in values[s]:
    return(values)
  values[s] = values[s].replace(d, '')
  if len(values[s]) == 0:
    return(False)
  elif len(values[s]) == 1:
    if not all(eliminate(values, s2, values[s]) for s2 in peers[s]):
      return(False)
  for u in units[s]:
    dplaces = [s for s in u if d in values[s]]
    if len(dplaces)==0:
      return(False)
    elif len(dplaces)==1 and len(values[dplaces[0]])>1:
      if not assign(values, dplaces[0], d):
        return(False)
  return(values)

def solve(grid):
  return(search(parse_grid(grid)))

def search(values):
  if values is False:
    return False
  if all(len(values[s])==1 for s in squares):
    return(values)
  n, s = min([len(values[s]), s] for s in squares if len(values[s])>1)
  """return some(search(assign(values.copy(), s, d)) 
        for d in values[s])"""
  for val in values[s]:
    solution = search(assign(values.copy(), s, val))
    if solution:
      return(solution)