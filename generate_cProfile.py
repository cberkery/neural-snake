import cProfile
from Run import run as snake_run

cProfile("snake_run(iterations=2)", filename="neural_snake_cprofile", sort=-1)

# cProfile.run(statement)
