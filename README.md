# CS5060_Program1

## Team Members
- Ann Marie Humble
- Brighton Ellis
- Nate Stott
- Madison Patch

## Optimal Stopping Results
### Part One
Our algorithm (meaning Dr. Harper's algorithm, thank you Dr. Harper) 
consistently found the most optimal candidates 
after searching around 37% of the candidate pool. The best strategy 
for choosing a candidate, then, would be to interview and dismiss
the first 37% of candidates, and then accept the next best candidate after that.
However, this strategy still only finds the optimal candidate about
37% percent of the time. Which isn't _comforting_. But it's the best we can do.

A side note, it's kind of cool that, with this brute-force solution, we get 37% for the
likelihood of finding the optimal candidate, as for the amount of candidates 
to go through before making a choice.

### Part Two

The optimal stopping point in a group of candidates with scores following a 
skewed beta distribution and in a group of candidates with scores following a normal distribution is very similar. 
The Beta(2,7) skewed distribution had a 35-41% chance of finding the optimal candidate after 
typically between 15-22 candidate evaluations. The normal distribution had a 37-41% chance of finding 
the optimal candidate after typically between 15-23 candidate evaluations. 
This demonstrates that the distribution shape does not strongly influence the optimal stopping point as long 
as there are roughly the same number of candidates between distributions that share a score.


### Part Three

For the normal distribution in part three, our graph showed that the optimal time to stop interviewing and start choosing was after candidate two, with around a 37% chance of success (assuming sample size of 50 candidats). We were initially very surprised by these results, but they made sense after we thought about it. In a normal distribution, the vast majority of candidates will have the same (or roughly the same) qualifications. With a penalty being administered after each additional interview, you very quickly reach a point where the chance of finding an exceptionally good candidate is outweighed by the time (penalty) it takes to find that candidate. The best strategy is to get a rough benchmark of the candidates and then pick quickly. What was shocking to us was the extent to which the left 

## Instructions to Run

1. install [poetry](https://python-poetry.org/docs/) on your machine
    1. prob just run this `pipx install poetry`
2. navigate to our project in the terminal
3. `poetry shell`
4. `poetry install`
5. `python3 explore_stopping.py`
