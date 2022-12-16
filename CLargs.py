import argparse


class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Optional arguments for run", prog="Run.py")
        parser.add_argument("-p", help="PopulationSize: Positive integer", required=False, type=int, default="")
        parser.add_argument("-i", help="Iterations: Positive integer", required=False, type=int, default="")
        parser.add_argument("-m", help="MutationRate: Float in (0,1)", required=False, type=float, default="")
        parser.add_argument("-s", help="SelectionProportion: Float in (0,1)", type=float, required=False, default="")

        argument = parser.parse_args()
        status = False

        # Set defaults
        self.p = 100
        self.i = 100
        self.m = 0.01
        self.s = 0.01

        if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
            status = True

        if argument.p:
            self.p = int(argument.p)
            print("You have used '--p' with argument: {0}".format(self.p))
            status = True

        if argument.i:
            self.i = argument.i
            print("You have used '--i' with argument: {0}".format(self.i))
            status = True

        if argument.m:
            self.m = argument.m
            print("You have used '--m' with argument: {0}".format(self.m))
            status = True

        if argument.s:
            self.s = argument.s
            print("You have used '-s' or '--s' with argument: {0}".format(self.s))
            status = True

        if not status:
            print("No arguments passed, running with defaults:")
            self.print_args()

    def print_args(self):
        print("p: {}".format(self.p))
        print("i:{}".format(self.i))
        print("Mutation_rate:{}".format(self.m))
        print("s:{}".format(self.s))

    def return_args(self):
        return self.p, self.i, self.m, self.s

    def print_set_defaults(self):
        pass


if __name__ == "__main__":
    app = CommandLine()
